from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Callable, Generator, List, Optional, Tuple

from langchain_core.documents import Document

from . import pipeline as _base
from .abbreviation import (
    build_abbreviation_regex,
    expand_abbreviations,
    load_abbreviation_dict,
)
from .config_seperate_pdb_id import (
    ABBR_DICT_PATH,
    ABBR_EXPANSION_ENABLED,
    ENRICH_CHEM_MAX_NAMES,
    ENRICH_RERANK_CHEMICAL,
    ENRICH_RERANK_STRUCTURE,
    ENRICH_RERANK_UNIPROT,
    ENRICH_UNIPROT_MAX_CHARS,
    INDEX_CONCAT_CHEMICAL,
    INDEX_CONCAT_METADATA,
    INDEX_CONCAT_STRUCTURE,
    INDEX_CONCAT_UNIPROT,
    INCLUDE_MISSING_PDB_AS_ABSTRACT_ONLY,
    LLM_BATCH_SIZE,
    LLM_MODEL_NAME,
    QUERY_DECOMPOSITION_ENABLED,
    QUERY_DECOMPOSITION_RRF_K,
    QUERY_REWRITE_ENABLED,
    RERANK_PDB_AGGREGATION,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_DISPLAY_K,
)


EntryTuple = _base.EntryTuple
AnswerAndEntriesSig = _base.AnswerAndEntriesSig
StreamingYield = _base.StreamingYield
AnswerAndEntriesStreamSig = _base.AnswerAndEntriesStreamSig

_answer_fn: AnswerAndEntriesSig | None = None
_streaming_fn: AnswerAndEntriesStreamSig | None = None


def _sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid for converting raw reranker scores to [0, 1].
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _score_bucket_label(normalized_score: float) -> str:
    if normalized_score < 0.25:
        return "[0,0.25)"
    if normalized_score < 0.5:
        return "[0.25,0.5)"
    if normalized_score < 0.75:
        return "[0.5,0.75)"
    return "[0.75,1]"


def _build_per_pdb_display_payload(
    pmid_to_pdb_raw_scores: dict[str, dict[str, List[float]]],
) -> dict[str, List[dict[str, Any]]]:
    """
    Convert per-PDB raw score lists into UI payloads.
    Uses max raw score per PDB for display while keeping ranking unchanged.
    """
    payload_by_pmid: dict[str, List[dict[str, Any]]] = {}
    for pmid, pdb_scores in pmid_to_pdb_raw_scores.items():
        items: List[dict[str, Any]] = []
        for pdb_id, raw_vals in pdb_scores.items():
            if not raw_vals:
                continue
            raw_score = float(max(raw_vals))
            normalized_score = _sigmoid(raw_score)
            items.append(
                {
                    "pdb_id": pdb_id,
                    "raw_score": raw_score,
                    "normalized_score": normalized_score,
                    "bucket_label": _score_bucket_label(normalized_score),
                }
            )
        if items:
            items.sort(key=lambda x: x["normalized_score"], reverse=True)
            payload_by_pmid[pmid] = items
    return payload_by_pmid


def __getattr__(name: str) -> Any:
    """
    Re-export all symbols from base pipeline unless explicitly overridden here.
    """
    return getattr(_base, name)


def _preprocess_query_for_decomposition(text: str) -> str:
    query = str(text).strip()
    query = query.replace("\nuser\n", " ")
    query = query.replace("user\n", " ")
    query = query.replace("structure:", " ")
    query = query.replace("structureCan you", "Can you")
    query = " ".join(query.split())
    return query


def _extract_json_block(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _parse_decomposition_output(
    raw_response: str,
    *,
    raw_query: str,
    cleaned_query: str,
) -> dict[str, Any]:
    fallback = {
        "Raw_query": raw_query,
        "need_decomposition": False,
        "sub_questions": [cleaned_query] if cleaned_query else [],
    }
    try:
        parsed = json.loads(_extract_json_block(raw_response))
        need_decomposition = bool(parsed.get("need_decomposition", False))
        sub_questions = parsed.get("sub_questions", [])
        if not isinstance(sub_questions, list):
            sub_questions = [str(sub_questions)]
        sub_questions = [str(x).strip() for x in sub_questions if str(x).strip()]
        if not sub_questions:
            return fallback
        if not need_decomposition:
            return {
                "Raw_query": raw_query,
                "need_decomposition": False,
                "sub_questions": [cleaned_query] if cleaned_query else sub_questions[:1],
            }
        return {
            "Raw_query": raw_query,
            "need_decomposition": True,
            "sub_questions": sub_questions,
        }
    except Exception:
        return fallback


def build_query_decomposition_chain(llm: Any) -> Any:
    prompt = _base.PromptTemplate.from_template(
        "You are helping with query decomposition for a structure-search RAG system.\n\n"
        "Rules:\n"
        "1) If the query is simple and asks one thing, set need_decomposition to false.\n"
        "2) If the query asks multiple sub-goals/comparisons/entities at once, set need_decomposition to true.\n"
        "3) Return valid JSON only, no markdown and no explanation.\n"
        "4) Output keys must be exactly: Raw_query, need_decomposition, sub_questions.\n"
        "5) sub_questions must be a list of strings.\n"
        "6) sub_questions must be a questions.\n"
        "7) If need_decomposition is false, sub_questions should contain one question.\n\n"
        "Raw query:\n{question}"
    )
    return prompt | llm | _base.StrOutputParser()


def decompose_query(chain: Any, raw_question: str) -> dict[str, Any]:
    cleaned_query = _preprocess_query_for_decomposition(raw_question)
    if not cleaned_query:
        return {
            "Raw_query": raw_question,
            "need_decomposition": False,
            "sub_questions": [],
        }
    raw_response = chain.invoke({"question": cleaned_query})
    if not isinstance(raw_response, str):
        raw_response = str(raw_response or "")
    return _parse_decomposition_output(
        raw_response,
        raw_query=raw_question,
        cleaned_query=cleaned_query,
    )


def aggregate_pmids_rrf(
    ranked_pmids_lists: List[List[str]],
    *,
    rrf_k: int,
    top_k: int,
) -> List[str]:
    if not ranked_pmids_lists:
        return []
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    c = max(1, int(rrf_k))
    for list_idx, ranked in enumerate(ranked_pmids_lists):
        seen_in_list: set[str] = set()
        for rank_idx, pmid in enumerate(ranked, start=1):
            pid = str(pmid).strip()
            if not pid or pid in seen_in_list:
                continue
            seen_in_list.add(pid)
            scores[pid] += 1.0 / (c + rank_idx)
            if pid not in first_seen:
                first_seen[pid] = (list_idx, rank_idx)
    ordered = sorted(scores.keys(), key=lambda p: (-scores[p], first_seen[p]))
    if top_k <= 0:
        return ordered
    return ordered[:top_k]


def _normalize_pdb_id(pdb_id: str) -> str:
    return (pdb_id or "").strip().upper()


def _records_for_pmid(
    pmid_to_rcsb: dict[str, List[dict]] | None, pmid: str
) -> List[dict]:
    """Resolve RCSB record list for a PMID (string keys; tolerate int-like keys)."""
    if not pmid_to_rcsb or not pmid:
        return []
    if pmid in pmid_to_rcsb:
        return pmid_to_rcsb[pmid]
    if pmid.isdigit():
        k = str(int(pmid))
        if k in pmid_to_rcsb:
            return pmid_to_rcsb[k]
    return []


def _select_records_for_pdb_ids(
    records: List[dict],
    pdb_ids: List[str],
) -> List[dict]:
    """
    Select RCSB records corresponding to retrieved pdb_ids, preserving pdb_ids order.
    """
    if not records:
        return []
    if not pdb_ids:
        return records

    by_pdb: dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        norm = _normalize_pdb_id(str(rec.get("pdb_id", "")))
        if norm:
            by_pdb[norm].append(rec)

    selected: List[dict] = []
    seen_ids: set[str] = set()
    for pdb_id in pdb_ids:
        norm = _normalize_pdb_id(pdb_id)
        if not norm or norm in seen_ids:
            continue
        seen_ids.add(norm)

        matches = by_pdb.get(norm, [])
        if matches:
            selected.extend(matches)
            continue

        if INCLUDE_MISSING_PDB_AS_ABSTRACT_ONLY:
            selected.append(
                {
                    "pdb_id": norm,
                    "title": "",
                    "keywords_text": "",
                    "keywords_pdbx": "",
                    "uniprot_functions": [],
                    "chem_components": [],
                }
            )

    if selected:
        return selected
    return records


def _build_single_record_sections(
    rec: dict,
    *,
    enrich_structure: bool,
    enrich_uniprot: bool,
    enrich_chemical: bool,
) -> tuple[str, str, str]:
    pdb_id = (rec.get("pdb_id") or "").strip()
    title = (rec.get("title") or "").strip()
    kw_text = (rec.get("keywords_text") or "").strip()
    kw_pdbx = (rec.get("keywords_pdbx") or "").strip()

    struct_parts: List[str] = []
    if enrich_structure:
        if pdb_id:
            struct_parts.append(f"PDB {pdb_id}")
        if title:
            struct_parts.append(title)
        if kw_text:
            struct_parts.append(f"Keywords: {kw_text}")
        if kw_pdbx:
            struct_parts.append(f"PDB keywords: {kw_pdbx}")
    struct_body = "; ".join(struct_parts).strip()

    uni_body = ""
    uni = rec.get("uniprot_functions") or []
    if enrich_uniprot and isinstance(uni, list) and uni:
        joined = _base._truncate_joined([str(x) for x in uni if x], ENRICH_UNIPROT_MAX_CHARS, sep=" | ")
        if joined:
            uni_body = f"PDB {pdb_id}: {joined}" if pdb_id else joined

    chem_body = ""

    chem = rec.get("chem_components") or []
    if enrich_chemical and isinstance(chem, list) and chem:
        shown = chem[:ENRICH_CHEM_MAX_NAMES]
        joined = ", ".join(str(x) for x in shown)
        if len(chem) > ENRICH_CHEM_MAX_NAMES:
            joined += f" … (+{len(chem) - ENRICH_CHEM_MAX_NAMES} more)"
        chem_body = f"PDB {pdb_id}: {joined}" if pdb_id else joined
    return struct_body, uni_body, chem_body


def _build_per_pdb_enriched_text(
    base_text: str,
    rec: dict,
    *,
    enrich_structure: bool,
    enrich_uniprot: bool,
    enrich_chemical: bool,
    max_chars: int = 10000,
) -> str:

    sections: List[str] = ["[Abstract]\n" + base_text.strip()]
    struct_b, uni_b, chem_b = _build_single_record_sections(
        rec,
        enrich_structure=enrich_structure,
        enrich_uniprot=enrich_uniprot,
        enrich_chemical=enrich_chemical,
    )
    if enrich_structure and struct_b:
        sections.append("[Structure info]\n" + struct_b)
    if enrich_uniprot and uni_b:
        sections.append("[UniProt]\n" + uni_b)

    if enrich_chemical and chem_b:
        sections.append("[Chemical components]\n" + chem_b)
    combined = "\n\n".join(sections)

    # print(combined)

    if len(combined) > max_chars:
        return combined[:max_chars]


    return combined


def _build_index_chunk_text(
    pmid: str,
    chunk_text: str,
    pmid_to_rcsb: dict[str, List[dict]] | None,
    *,
    include_structure: bool,
    include_uniprot: bool,
    include_chemical: bool,
    max_chars: int = 10000,
) -> str:
    """
    Build indexing text in the exact format:
    [Abstract]<chunk> + selected metadata sections.
    """
    if not pmid_to_rcsb:
        return chunk_text

    struct_b, uni_b, chem_b = _base._collect_rcsb_section_bodies(
        pmid,
        pmid_to_rcsb,
        enrich_uniprot=include_uniprot,
        enrich_chemical=include_chemical,
    )

    sections: List[str] = ["[Abstract]\n" + chunk_text.strip()]
    if include_structure and struct_b:
        sections.append("[Structure info]\n" + struct_b)
    if include_uniprot and uni_b:
        sections.append("[UniProt]\n" + uni_b)
    if include_chemical and chem_b:
        sections.append("[Chemical components]\n" + chem_b)

    if len(sections) == 1:
        return chunk_text
    combined = "\n\n".join(sections)
    if len(combined) > max_chars:
        return combined[:max_chars]
    return combined


def maybe_enrich_splits_for_indexing(
    splits: List[Document],
    pmid_to_rcsb: dict[str, List[dict]] | None,
    *,
    concat_metadata: bool | None = None,
    include_structure: bool | None = None,
    include_uniprot: bool | None = None,
    include_chemical: bool | None = None,
) -> List[Document]:
    """
    Optionally enrich each chunk before indexing so embeddings represent
    [Abstract chunk] + selected metadata sections.
    """
    cm = INDEX_CONCAT_METADATA if concat_metadata is None else concat_metadata
    if not cm:
        return splits

    istruct = INDEX_CONCAT_STRUCTURE if include_structure is None else include_structure
    iuni = INDEX_CONCAT_UNIPROT if include_uniprot is None else include_uniprot
    ichem = INDEX_CONCAT_CHEMICAL if include_chemical is None else include_chemical

    enriched: List[Document] = []
    for split in splits:
        pmid = str(split.metadata.get("PubMed_ID", "")).strip()
        if not pmid:
            enriched.append(split)
            continue
        text = _build_index_chunk_text(
            pmid,
            split.page_content,
            pmid_to_rcsb,
            include_structure=istruct,
            include_uniprot=iuni,
            include_chemical=ichem,
        )
        enriched.append(Document(page_content=text, metadata=dict(split.metadata)))
    return enriched


def rerank_entries(
    query: str,
    entries: List[Tuple[str, str, List[str]]],
    top_k: int,
    pmid_to_fulltext: dict[str, str] | None = None,
    model_name: str | None = None,
    pmid_to_rcsb: dict[str, List[dict]] | None = None,
    *,
    enrich_structure: bool | None = None,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
) -> List[Tuple[str, str, List[str]]]:
    """
    Per-PDB reranking variant:
    - Build one cross-encoder candidate per (PMID, PDB record)
    - Score all candidates together
    - Collapse scores back to PMID using configured aggregation
    """
    if not entries:
        return []

    es = ENRICH_RERANK_STRUCTURE if enrich_structure is None else enrich_structure
    eu = ENRICH_RERANK_UNIPROT if enrich_uniprot is None else enrich_uniprot
    ec = ENRICH_RERANK_CHEMICAL if enrich_chemical is None else enrich_chemical
    reranker = _base.build_reranker(model_name or RERANKER_MODEL)
    reranker.max_seq_length = 4096

    entry_by_pmid: dict[str, Tuple[str, str, List[str]]] = {}
    entry_index: dict[str, int] = {}
    pair_pmids: List[str] = []
    pair_texts: List[str] = []

    for i, (pmid, abstract, pdb_ids) in enumerate(entries):
        base_text = abstract
        if pmid_to_fulltext:
            full = pmid_to_fulltext.get(pmid)
            if full:
                base_text = full

        if pmid not in entry_by_pmid:
            entry_by_pmid[pmid] = (pmid, base_text, pdb_ids)
            entry_index[pmid] = i

        records = _records_for_pmid(pmid_to_rcsb, pmid)
        selected_records = _select_records_for_pdb_ids(records, pdb_ids)
        if selected_records:
            for rec in selected_records:
                pair_pmids.append(pmid)
                pair_texts.append(
                    _build_per_pdb_enriched_text(
                        base_text,
                        rec,
                        enrich_structure=es,
                        enrich_uniprot=eu,
                        enrich_chemical=ec,
                    )
                )
            continue

        pair_pmids.append(pmid)
        pair_texts.append(base_text)

    if not pair_texts:
        return []

    pairs = [(query, text) for text in pair_texts]
    scores = reranker.predict(pairs)

    score_lists: dict[str, List[float]] = defaultdict(list)
    for pmid, score in zip(pair_pmids, scores, strict=False):
        score_lists[pmid].append(float(score))

    agg_scores: dict[str, float] = {}
    for pmid, vals in score_lists.items():
        if not vals:
            continue
        if RERANK_PDB_AGGREGATION == "mean":
            agg_scores[pmid] = sum(vals) / len(vals)
        else:
            agg_scores[pmid] = max(vals)

    ordered_pmids = sorted(
        agg_scores.keys(),
        key=lambda p: (-agg_scores[p], entry_index.get(p, 10**9)),
    )
    return [entry_by_pmid[p] for p in ordered_pmids[:top_k] if p in entry_by_pmid]


def rerank_entries_with_per_pdb_scores(
    query: str,
    entries: List[Tuple[str, str, List[str]]],
    top_k: int,
    pmid_to_fulltext: dict[str, str] | None = None,
    model_name: str | None = None,
    pmid_to_rcsb: dict[str, List[dict]] | None = None,
    *,
    enrich_structure: bool | None = None,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
) -> tuple[List[Tuple[str, str, List[str]]], dict[str, List[dict[str, Any]]]]:
    """
    Per-PDB reranking with score side-channel for UI display.
    Ranking behavior remains identical to rerank_entries (raw score aggregation).
    """
    if not entries:
        return [], {}

    es = ENRICH_RERANK_STRUCTURE if enrich_structure is None else enrich_structure
    eu = ENRICH_RERANK_UNIPROT if enrich_uniprot is None else enrich_uniprot
    ec = ENRICH_RERANK_CHEMICAL if enrich_chemical is None else enrich_chemical
    reranker = _base.build_reranker(model_name or RERANKER_MODEL)
    reranker.max_seq_length = 4096

    entry_by_pmid: dict[str, Tuple[str, str, List[str]]] = {}
    entry_index: dict[str, int] = {}
    pair_pmids: List[str] = []
    pair_pdb_ids: List[str] = []
    pair_texts: List[str] = []
    pmid_to_pdb_raw_scores: dict[str, dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for i, (pmid, abstract, pdb_ids) in enumerate(entries):
        base_text = abstract
        if pmid_to_fulltext:
            full = pmid_to_fulltext.get(pmid)
            if full:
                base_text = full

        if pmid not in entry_by_pmid:
            entry_by_pmid[pmid] = (pmid, base_text, pdb_ids)
            entry_index[pmid] = i

        records = _records_for_pmid(pmid_to_rcsb, pmid)
        selected_records = _select_records_for_pdb_ids(records, pdb_ids)
        if selected_records:
            for rec in selected_records:
                pair_pmids.append(pmid)
                pair_pdb_ids.append(
                    _normalize_pdb_id(str(rec.get("pdb_id", ""))) or "UNKNOWN"
                )
                pair_texts.append(
                    _build_per_pdb_enriched_text(
                        base_text,
                        rec,
                        enrich_structure=es,
                        enrich_uniprot=eu,
                        enrich_chemical=ec,
                    )
                )
            continue

        pair_pmids.append(pmid)
        pair_pdb_ids.append(_normalize_pdb_id(pdb_ids[0]) if pdb_ids else "UNKNOWN")
        pair_texts.append(base_text)

    if not pair_texts:
        return [], {}

    pairs = [(query, text) for text in pair_texts]
    scores = reranker.predict(pairs)

    score_lists: dict[str, List[float]] = defaultdict(list)
    for pmid, pdb_id, score in zip(pair_pmids, pair_pdb_ids, scores, strict=False):
        numeric_score = float(score)
        score_lists[pmid].append(numeric_score)
        pmid_to_pdb_raw_scores[pmid][pdb_id].append(numeric_score)

    agg_scores: dict[str, float] = {}
    for pmid, vals in score_lists.items():
        if not vals:
            continue
        if RERANK_PDB_AGGREGATION == "mean":
            agg_scores[pmid] = sum(vals) / len(vals)
        else:
            agg_scores[pmid] = max(vals)

    ordered_pmids = sorted(
        agg_scores.keys(),
        key=lambda p: (-agg_scores[p], entry_index.get(p, 10**9)),
    )
    ordered_entries = [entry_by_pmid[p] for p in ordered_pmids[:top_k] if p in entry_by_pmid]
    per_pdb_payload = _build_per_pdb_display_payload(pmid_to_pdb_raw_scores)
    return ordered_entries, per_pdb_payload


def run_retrieve_rerank(
    question: str,
    retriever: Any,
    pmid_to_abstract: dict[str, str],
    pmid_to_pdb: dict[str, List[str]],
    top_k: int = 50,
    pmid_to_rcsb: dict[str, List[dict]] | None = None,
    *,
    enrich_structure: bool | None = None,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
    pmid_to_fulltext: dict[str, str] | None = None,
    model_name: str | None = None,
) -> List[str]:
    """
    Retrieve → ``rerank_entries`` (per-PDB variant in this module) → top-k PMIDs.

    Pass ``enrich_structure`` / ``enrich_uniprot`` / ``enrich_chemical`` explicitly for
    ablations; ``None`` uses ``config_seperate_pdb_id`` defaults.
    """
    question = (question or "").strip()
    if not question:
        return []
    docs_retrieved = retriever.get_relevant_documents(question)
    raw_entries: List[Tuple[str, str, List[str]]] = []
    for doc in docs_retrieved:
        pmid = str(doc.metadata.get("PubMed_ID", "?"))
        full_abstract = pmid_to_abstract.get(pmid, doc.page_content)
        pdb_ids = pmid_to_pdb.get(pmid, [])
        raw_entries.append((pmid, full_abstract, pdb_ids))
    if not raw_entries:
        return []
    reranked = rerank_entries(
        question,
        raw_entries,
        top_k=top_k,
        pmid_to_fulltext=pmid_to_fulltext,
        model_name=model_name,
        pmid_to_rcsb=pmid_to_rcsb,
        enrich_structure=enrich_structure,
        enrich_uniprot=enrich_uniprot,
        enrich_chemical=enrich_chemical,
    )
    return [e[0] for e in reranked]


def run_retrieve_then_rerank_with_other_query(
    retrieval_question: str,
    rerank_question: str,
    retriever: Any,
    pmid_to_abstract: dict[str, str],
    pmid_to_pdb: dict[str, List[str]],
    top_k: int = 50,
    pmid_to_rcsb: dict[str, List[dict]] | None = None,
    *,
    enrich_structure: bool | None = None,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
    pmid_to_fulltext: dict[str, str] | None = None,
    model_name: str | None = None,
) -> List[str]:
    """Retrieve with ``retrieval_question``, then rerank candidates with ``rerank_question``."""
    retrieval_question = (retrieval_question or "").strip()
    rerank_question = (rerank_question or "").strip() or retrieval_question
    if not retrieval_question:
        return []
    docs_retrieved = retriever.get_relevant_documents(retrieval_question)
    raw_entries: List[Tuple[str, str, List[str]]] = []
    for doc in docs_retrieved:
        pmid = str(doc.metadata.get("PubMed_ID", "?"))
        full_abstract = pmid_to_abstract.get(pmid, doc.page_content)
        pdb_ids = pmid_to_pdb.get(pmid, [])
        raw_entries.append((pmid, full_abstract, pdb_ids))
    if not raw_entries:
        return []
    reranked = rerank_entries(
        rerank_question,
        raw_entries,
        top_k=top_k,
        pmid_to_fulltext=pmid_to_fulltext,
        model_name=model_name,
        pmid_to_rcsb=pmid_to_rcsb,
        enrich_structure=enrich_structure,
        enrich_uniprot=enrich_uniprot,
        enrich_chemical=enrich_chemical,
    )
    return [e[0] for e in reranked]


def run_retrieve_rerank_reason_judge(
    question: str,
    retriever: Any,
    pmid_to_abstract: dict[str, str],
    pmid_to_pdb: dict[str, List[str]],
    reason_chain: Any,
    judge_chain: Any,
    top_k: int,
    pmid_to_rcsb: dict[str, List[dict]] | None = None,
    pmid_to_fulltext: dict[str, str] | None = None,
    llm_batch_size: int | None = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    query_rewrite_chain: Any | None = None,
    query_decomposition_chain: Any | None = None,
    skip_query_rewrite: bool = False,
    use_query_decomposition: bool | None = None,
    abbr_dict: dict[str, str] | None = None,
    abbr_regex: Any | None = None,
    *,
    enrich_structure: bool | None = None,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
) -> Tuple[List[str], List[EntryTuple]]:
    cleaned_question = _base._prepare_query_for_retrieval(
        question, query_rewrite_chain, skip_query_rewrite
    )
    if not cleaned_question:
        return [], []

    if abbr_dict and abbr_regex is not None:
        expanded_question, replacements = expand_abbreviations(
            cleaned_question, abbr_dict, abbr_regex
        )
        if replacements:
            summary = ", ".join(
                f"{r['abbr']}->{r['full_name']}" for r in replacements
            )
            if progress_callback:
                progress_callback(
                    0.03,
                    f"Expanded {len(replacements)} abbreviation(s): {summary}",
                )
            cleaned_question = expanded_question

    use_decomp = (
        QUERY_DECOMPOSITION_ENABLED
        if use_query_decomposition is None
        else use_query_decomposition
    )

    raw_entries: List[Tuple[str, str, List[str]]] = []
    per_pdb_payload: dict[str, List[dict[str, Any]]] = {}
    if use_decomp and query_decomposition_chain is not None:
        if progress_callback:
            progress_callback(0.05, "Determining whether query needs decomposition...")
        decomp = decompose_query(query_decomposition_chain, cleaned_question)
        need_decomposition = bool(decomp.get("need_decomposition", False))
        if need_decomposition:
            sub_questions = [
                str(q).strip()
                for q in (decomp.get("sub_questions") or [])
                if str(q).strip()
            ]
            if sub_questions:
                if progress_callback:
                    progress_callback(
                        0.1,
                        f"Running retrieval/rerank for {len(sub_questions)} decomposed sub-questions...",
                    )
                ranked_lists: List[List[str]] = []
                for idx, sub_q in enumerate(sub_questions, start=1):
                    if progress_callback:
                        progress_callback(
                            0.1,
                            f"Retrieving with sub-question {idx}/{len(sub_questions)} and reranking with the original question...",
                        )
                    ranked_lists.append(
                        run_retrieve_then_rerank_with_other_query(
                            retrieval_question=sub_q,
                            rerank_question=cleaned_question,
                            retriever=retriever,
                            pmid_to_abstract=pmid_to_abstract,
                            pmid_to_pdb=pmid_to_pdb,
                            top_k=top_k,
                            pmid_to_rcsb=pmid_to_rcsb,
                            enrich_structure=enrich_structure,
                            enrich_uniprot=enrich_uniprot,
                            enrich_chemical=enrich_chemical,
                            pmid_to_fulltext=pmid_to_fulltext,
                        )
                    )
                if progress_callback:
                    progress_callback(0.15, "Aggregating sub-question reranked results...")
                aggregated_pmids = aggregate_pmids_rrf(
                    ranked_lists, rrf_k=QUERY_DECOMPOSITION_RRF_K, top_k=top_k
                )
                raw_entries = [
                    (
                        pmid,
                        (
                            pmid_to_fulltext.get(pmid, "")
                            if pmid_to_fulltext
                            else pmid_to_abstract.get(pmid, "")
                        )
                        or pmid_to_abstract.get(pmid, ""),
                        pmid_to_pdb.get(pmid, []),
                    )
                    for pmid in aggregated_pmids
                ]

    if not raw_entries:
        if progress_callback:
            progress_callback(0.1, "Retrieving relevant documents...")
        docs_retrieved = retriever.get_relevant_documents(cleaned_question)

        for doc in docs_retrieved:
            pmid = str(doc.metadata.get("PubMed_ID", "?"))
            full_abstract = pmid_to_abstract.get(pmid, doc.page_content)
            pdb_ids = pmid_to_pdb.get(pmid, [])
            raw_entries.append((pmid, full_abstract, pdb_ids))

        if progress_callback:
            progress_callback(0.15, "Reranking retrieved articles...")
        raw_entries, per_pdb_payload = rerank_entries_with_per_pdb_scores(
            cleaned_question,
            raw_entries,
            top_k=top_k,
            pmid_to_fulltext=pmid_to_fulltext,
            pmid_to_rcsb=pmid_to_rcsb,
            enrich_structure=enrich_structure,
            enrich_uniprot=enrich_uniprot,
            enrich_chemical=enrich_chemical,
        )

    chunk = llm_batch_size if llm_batch_size is not None else LLM_BATCH_SIZE
    entries = _base._entries_after_reason_judge(
        cleaned_question,
        raw_entries,
        reason_chain,
        judge_chain,
        chunk,
        progress_callback=progress_callback,
    )
    if per_pdb_payload:
        entries = [
            (*entry, per_pdb_payload.get(entry[0], []))
            for entry in entries
        ]
    if progress_callback:
        progress_callback(1.0, "Done.")
    return [e[0] for e in entries], entries


def initialize_pipeline() -> Tuple[AnswerAndEntriesSig, AnswerAndEntriesStreamSig]:
    """
    Initialize a side-by-side pipeline variant with per-PDB reranking.
    """
    global _answer_fn, _streaming_fn
    if _answer_fn is not None and _streaming_fn is not None:
        return _answer_fn, _streaming_fn

    print("[RAG] Loading documents from JSON...")
    docs = _base.load_documents()

    print("[RAG] Loading PMID -> full abstract, PMID -> PDB IDs, and RCSB metadata lookups...")
    pmid_to_abstract = _base.load_pmid_to_abstract()
    pmid_to_pdb = _base.load_pmid_to_pdb_ids()
    pmid_to_rcsb = _base.load_pmid_to_rcsb_metadata()

    print(
        "[RAG] Reranker enrichment flags: "
        f"structure={ENRICH_RERANK_STRUCTURE}, "
        f"UniProt={ENRICH_RERANK_UNIPROT}, chemical={ENRICH_RERANK_CHEMICAL} "
        "(set RAD_RAG_ENRICH_STRUCTURE / RAD_RAG_ENRICH_UNIPROT / "
        "RAD_RAG_ENRICH_CHEMICAL to 0 or 1 to compare)"
    )
    print(f"[RAG] Per-PDB aggregation: {RERANK_PDB_AGGREGATION}")
    print(
        "[RAG] Index metadata concat flags: "
        f"enabled={INDEX_CONCAT_METADATA}, "
        f"structure={INDEX_CONCAT_STRUCTURE}, "
        f"UniProt={INDEX_CONCAT_UNIPROT}, chemical={INDEX_CONCAT_CHEMICAL}"
    )

    print(f"[RAG] Loaded {len(docs)} documents. Building splits...")
    splits = _base.build_splits(docs)
    splits_for_index = maybe_enrich_splits_for_indexing(splits, pmid_to_rcsb)

    print(f"[RAG] Built {len(splits_for_index)} chunks. Creating vectorstore...")
    vectorstore = _base.build_vectorstore(splits_for_index, persist=True)

    print("[RAG] Creating hybrid retriever (dense + BM25)...")
    retriever = _base.build_retriever(vectorstore, splits_for_index, k=RETRIEVAL_DISPLAY_K)

    print(f"[RAG] Loading reranker: {RERANKER_MODEL}...")
    _base.build_reranker(RERANKER_MODEL)

    print(f"[RAG] Loading LLM: {LLM_MODEL_NAME}...")
    llm = _base.build_llm()

    print("[RAG] Building reason chain (per-article relevance)...")
    reason_chain = _base.build_reason_chain(llm)
    print("[RAG] Building judge chain (reranking, LLM-as-judge)...")
    judge_chain = _base.build_judge_chain(llm)

    query_rewrite_chain: Any | None = None
    query_decomposition_chain: Any | None = None
    if QUERY_REWRITE_ENABLED:
        print("[RAG] Building query rewrite chain (pre-retrieval)...")
        query_rewrite_chain = _base.build_query_rewrite_chain(llm)
    if QUERY_DECOMPOSITION_ENABLED:
        print("[RAG] Building query decomposition chain (pre-retrieval decision)...")
        query_decomposition_chain = build_query_decomposition_chain(llm)

    abbr_dict: dict[str, str] = {}
    abbr_regex: Any | None = None
    if ABBR_EXPANSION_ENABLED:
        if ABBR_DICT_PATH.exists():
            print(f"[RAG] Loading abbreviation dictionary from {ABBR_DICT_PATH}...")
            abbr_dict = load_abbreviation_dict(ABBR_DICT_PATH)
            abbr_regex = build_abbreviation_regex(abbr_dict)
            print(
                f"[RAG] Abbreviation expansion enabled: "
                f"{len(abbr_dict)} unambiguous abbreviations loaded."
            )
        else:
            print(
                f"[RAG] Abbreviation expansion enabled but dictionary not found "
                f"at {ABBR_DICT_PATH}; skipping."
            )
    else:
        print("[RAG] Abbreviation expansion disabled.")

    def answer_and_entries(
        question: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        use_query_decomposition: bool | None = None,
    ) -> Tuple[str, List[EntryTuple]]:
        if not question.strip():
            return "Please provide a non-empty question.", []
        _pmids, all_with_judgement = run_retrieve_rerank_reason_judge(
            question,
            retriever,
            pmid_to_abstract,
            pmid_to_pdb,
            reason_chain,
            judge_chain,
            top_k=RERANK_TOP_K,
            pmid_to_rcsb=pmid_to_rcsb,
            pmid_to_fulltext=None,
            llm_batch_size=LLM_BATCH_SIZE,
            progress_callback=progress_callback,
            query_rewrite_chain=query_rewrite_chain,
            query_decomposition_chain=query_decomposition_chain,
            use_query_decomposition=use_query_decomposition,
            abbr_dict=abbr_dict,
            abbr_regex=abbr_regex,
        )
        n = len(all_with_judgement)
        answer = (
            f"Retrieved articles; showing the top {n} sorted by relevance score (0-2), with ties in reranker order. "
            "Below: each article's PMID, PDB IDs, reason, judgement, score, and full abstract (relevant entries highlighted)."
        )
        return answer, all_with_judgement

    def answer_and_entries_streaming(
        question: str,
        use_query_decomposition: bool | None = None,
    ) -> Generator[StreamingYield, None, None]:
        if not question.strip():
            yield ("result", "Please provide a non-empty question.", [])
            return
        yield ("status", "Running retrieval and reranking...")
        _pmids, all_with_judgement = run_retrieve_rerank_reason_judge(
            question,
            retriever,
            pmid_to_abstract,
            pmid_to_pdb,
            reason_chain,
            judge_chain,
            top_k=RERANK_TOP_K,
            pmid_to_rcsb=pmid_to_rcsb,
            pmid_to_fulltext=None,
            llm_batch_size=LLM_BATCH_SIZE,
            query_rewrite_chain=query_rewrite_chain,
            query_decomposition_chain=query_decomposition_chain,
            use_query_decomposition=use_query_decomposition,
            abbr_dict=abbr_dict,
            abbr_regex=abbr_regex,
        )
        n = len(all_with_judgement)
        yield ("status", "Done.")
        answer = (
            f"Retrieved articles; showing the top {n} sorted by relevance score (0-2), with ties in reranker order. "
            "Below: each article's PMID, PDB IDs, reason, judgement, score, and full abstract (relevant entries highlighted)."
        )
        yield ("result", answer, all_with_judgement)

    _answer_fn = answer_and_entries
    _streaming_fn = answer_and_entries_streaming
    print("[RAG] Pipeline initialization complete (per-PDB rerank variant).")
    return _answer_fn, _streaming_fn
