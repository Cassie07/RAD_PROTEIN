from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple

from langchain import hub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

from .config import (
    CHROMA_DIR,
    DATA_PATH,
    EMBEDDING_MODEL_NAME,
    ENRICH_CHEM_MAX_NAMES,
    ENRICH_RERANK_CHEMICAL,
    ENRICH_RERANK_UNIPROT,
    ENRICH_UNIPROT_MAX_CHARS,
    LLM_BATCH_SIZE,
    LLM_MODEL_NAME,
    QUERY_REWRITE_ENABLED,
    RCSB_DATA_PATH,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_DISPLAY_K,
    TOP_K,
)


def load_documents(path: Path | str | None = None) -> List[Document]:
    """
    Load PubMed-style documents from JSON and convert them into LangChain Documents.

    Mirrors the notebook logic:
      - reads demo_pubmed.json
      - deduplicates by Abstract content
      - stores PubMed_ID in metadata
    """
    json_path = Path(path) if path is not None else DATA_PATH
    if not json_path.exists():
        raise FileNotFoundError(f"Data file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    docs: List[Document] = []
    seen_abstracts: set[str] = set()

    for entry in entries:
        title = entry.get("PubMed_ID", "")
        abstract = entry.get("Abstract", "")
        if not abstract or abstract in seen_abstracts:
            continue
        seen_abstracts.add(abstract)
        docs.append(
            Document(page_content=abstract, metadata={"PubMed_ID": title})
        )

    if not docs:
        raise ValueError("No documents were loaded from the JSON file.")

    return docs


def load_pmid_to_abstract(path: Path | str | None = None) -> dict[str, str]:
    """
    Build PMID -> full abstract from the same JSON as load_documents (DATA_PATH).
    Used to show entire abstract per retrieved PMID in the UI.
    """
    json_path = Path(path) if path is not None else DATA_PATH
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    out: dict[str, str] = {}
    for entry in entries:
        pmid = entry.get("PubMed_ID")
        abstract = entry.get("Abstract", "")
        if pmid is not None and abstract:
            out[str(pmid)] = abstract
    return out


def load_pmid_to_pdb_ids(path: Path | str | None = None) -> dict[str, List[str]]:
    """
    Build PMID -> list of PDB (RCSB) IDs from full_rcsb_data_2026.json.
    Mirrors 1. Single_abs_question_generation_2026.ipynb (dict_rid_pid then dict_pid_rid).
    Returns dict keyed by str(pubmed_id) for consistency with RAG metadata.
    """
    json_path = Path(path) if path is not None else RCSB_DATA_PATH
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    entries = data.get("data", {}).get("entries", [])
    dict_rid_pid: dict[str, int] = {}
    for entry in entries:
        try:
            pubmed = entry.get("pubmed")
            if not pubmed:
                continue
            pubmed_id = pubmed["rcsb_pubmed_container_identifiers"]["pubmed_id"]
            rcsb_id = entry.get("rcsb_id")
            if rcsb_id is not None:
                dict_rid_pid[rcsb_id] = pubmed_id
        except (KeyError, TypeError):
            continue
    dict_pid_rid: dict[str, List[str]] = defaultdict(list)
    for rid, pid in dict_rid_pid.items():
        dict_pid_rid[str(pid)].append(rid)
    return dict(dict_pid_rid)


def _entry_uniprot_function_texts(entry: dict) -> List[str]:
    """Collect unique UniProt function.details strings from polymer_entities."""
    texts: List[str] = []
    for pe in entry.get("polymer_entities") or []:
        if not isinstance(pe, dict):
            continue
        for u in pe.get("uniprots") or []:
            if not isinstance(u, dict):
                continue
            protein = u.get("rcsb_uniprot_protein") or {}
            fn = protein.get("function") or {}
            det = fn.get("details")
            if isinstance(det, str):
                s = det.strip()
                if s:
                    texts.append(s)
    seen: set[str] = set()
    out: List[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _entry_chem_component_names(entry: dict) -> List[str]:
    """Collect chemical component and synonym names from nonpolymer_entities."""
    names: List[str] = []
    for np in entry.get("nonpolymer_entities") or []:
        if not isinstance(np, dict):
            continue
        npc = np.get("nonpolymer_comp") or {}
        if not isinstance(npc, dict):
            continue
        cc = npc.get("chem_comp") or {}
        if isinstance(cc, dict):
            n = cc.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
        for sy in npc.get("rcsb_chem_comp_synonyms") or []:
            if isinstance(sy, dict):
                sn = sy.get("name")
                if isinstance(sn, str) and sn.strip():
                    names.append(sn.strip())
    seen: set[str] = set()
    out: List[str] = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def load_pmid_to_rcsb_metadata(path: Path | str | None = None) -> dict[str, List[dict]]:
    """
    Build PMID -> list of RCSB/PDB metadata dicts from full_rcsb_data_2026.json.

    Each metadata dict is compact and may include (from full_rcsb_data JSON):
        pdb_id, title, keywords_text, keywords_pdbx,
        uniprot_functions (list of function.details strings),
        chem_components (deduped ligand / small-molecule names and synonyms).
    """
    json_path = Path(path) if path is not None else RCSB_DATA_PATH
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    entries = data.get("data", {}).get("entries", [])
    pmid_to_rcsb: dict[str, List[dict]] = defaultdict(list)

    for entry in entries:
        try:
            pubmed = entry.get("pubmed")
            if not pubmed:
                continue
            rcsb_id = entry.get("rcsb_id") or ""
            pubmed_id = pubmed["rcsb_pubmed_container_identifiers"]["pubmed_id"]
            if pubmed_id is None:
                continue

            struct = entry.get("struct", {}) or {}
            struct_keywords = entry.get("struct_keywords", {}) or {}

            title = struct.get("title") or ""
            keywords_text = struct_keywords.get("text") or ""
            keywords_pdbx = struct_keywords.get("pdbx_keywords") or ""

            uniprot_functions = _entry_uniprot_function_texts(entry)
            chem_components = _entry_chem_component_names(entry)

            if not any(
                [title, keywords_text, keywords_pdbx, uniprot_functions, chem_components]
            ):
                continue

            pmid_to_rcsb[str(pubmed_id)].append(
                {
                    "pdb_id": rcsb_id,
                    "title": title,
                    "keywords_text": keywords_text,
                    "keywords_pdbx": keywords_pdbx,
                    "uniprot_functions": uniprot_functions,
                    "chem_components": chem_components,
                }
            )
        except (KeyError, TypeError):
            continue

    return dict(pmid_to_rcsb)


def _truncate_joined(texts: List[str], max_total: int, sep: str = " | ") -> str:
    """Join non-empty strings with sep; truncate with ellipsis if longer than max_total."""
    bits = [str(t).strip() for t in texts if t and str(t).strip()]
    if not bits:
        return ""
    s = sep.join(bits)
    if len(s) <= max_total:
        return s
    if max_total < 2:
        return ""
    return s[: max_total - 1] + "…"


def _collect_rcsb_section_bodies(
    pmid: str,
    pmid_to_rcsb: dict[str, List[dict]],
    max_records: int = 10,
    *,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
    uniprot_max_chars: int | None = None,
    chem_max_names: int | None = None,
) -> tuple[str, str, str]:
    """
    Build three body strings (no bracket headers) for structure, UniProt, and chemical
    sections from RCSB metadata for a PMID. Empty string if a section has no content.
    """
    records = pmid_to_rcsb.get(pmid, [])
    if not records:
        return "", "", ""

    u_cap = uniprot_max_chars if uniprot_max_chars is not None else ENRICH_UNIPROT_MAX_CHARS
    n_cap = chem_max_names if chem_max_names is not None else ENRICH_CHEM_MAX_NAMES
    eu = ENRICH_RERANK_UNIPROT if enrich_uniprot is None else enrich_uniprot
    ec = ENRICH_RERANK_CHEMICAL if enrich_chemical is None else enrich_chemical

    detailed_records = records[:max_records] if max_records is not None else records

    struct_lines: List[str] = []
    uni_lines: List[str] = []
    chem_lines: List[str] = []

    for rec in detailed_records:
        pdb_id = (rec.get("pdb_id") or "").strip()
        title = (rec.get("title") or "").strip()
        kw_text = (rec.get("keywords_text") or "").strip()
        kw_pdbx = (rec.get("keywords_pdbx") or "").strip()

        parts: List[str] = []
        if pdb_id:
            parts.append(f"PDB {pdb_id}")
        if title:
            parts.append(title)
        if kw_text:
            parts.append(f"Keywords: {kw_text}")
        if kw_pdbx:
            parts.append(f"PDB keywords: {kw_pdbx}")

        if parts:
            struct_lines.append("; ".join(parts))

        uni = rec.get("uniprot_functions") or []
        if eu and isinstance(uni, list) and uni:
            u_text = _truncate_joined([str(x) for x in uni if x], u_cap, sep=" | ")
            if u_text:
                if pdb_id:
                    uni_lines.append(f"PDB {pdb_id}: {u_text}")
                else:
                    uni_lines.append(u_text)

        chem = rec.get("chem_components") or []
        if ec and isinstance(chem, list) and chem:
            shown = chem[:n_cap]
            chem_body = ", ".join(str(x) for x in shown)
            if len(chem) > n_cap:
                chem_body += f" … (+{len(chem) - n_cap} more)"
            if pdb_id:
                chem_lines.append(f"PDB {pdb_id}: {chem_body}")
            else:
                chem_lines.append(chem_body)

    return (
        "\n".join(struct_lines).strip(),
        "\n".join(uni_lines).strip(),
        "\n".join(chem_lines).strip(),
    )


# def _build_rcsb_snippet(
#     pmid: str,
#     pmid_to_rcsb: dict[str, List[dict]],
#     max_records: int = 10,
#     *,
#     enrich_uniprot: bool | None = None,
#     enrich_chemical: bool | None = None,
#     uniprot_max_chars: int | None = None,
#     chem_max_names: int | None = None,
# ) -> str:
#     """
#     Flat snippet (legacy): structure, UniProt, and chemical lines interleaved.
#     Prefer ``enrich_text_with_rcsb`` for the bracketed multi-section format.
#     """
#     struct_b, uni_b, chem_b = _collect_rcsb_section_bodies(
#         pmid,
#         pmid_to_rcsb,
#         max_records=max_records,
#         enrich_uniprot=enrich_uniprot,
#         enrich_chemical=enrich_chemical,
#         uniprot_max_chars=uniprot_max_chars,
#         chem_max_names=chem_max_names,
#     )
#     eu = ENRICH_RERANK_UNIPROT if enrich_uniprot is None else enrich_uniprot
#     ec = ENRICH_RERANK_CHEMICAL if enrich_chemical is None else enrich_chemical
#     chunks: List[str] = []
#     if struct_b:
#         chunks.append(struct_b)
#     if eu and uni_b:
#         chunks.append(uni_b)
#     if ec and chem_b:
#         chunks.append(chem_b)
#     return "\n".join(chunks)


def enrich_text_with_rcsb(
    pmid: str,
    text: str,
    pmid_to_rcsb: dict[str, List[dict]] | None,
    max_chars: int = 10000,
    *,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
) -> str:
    """
    Combine the base text (abstract or full text) with RCSB-derived sections.

    Uses the same layout for every included block::

        [Abstract]
        <text>

        [Structure info]
        <structure lines>

        [UniProt]
        <function text>   (when enrich_uniprot is true and data exists)

        [Chemical components]
        <names>           (when enrich_chemical is true and data exists)

    Sections are separated by ``\\n\\n``; omitted sections are skipped. If no RCSB
    metadata applies, the original ``text`` is returned unchanged.
    """
    if not pmid_to_rcsb:
        return text

    eu = ENRICH_RERANK_UNIPROT if enrich_uniprot is None else enrich_uniprot
    ec = ENRICH_RERANK_CHEMICAL if enrich_chemical is None else enrich_chemical

    struct_b, uni_b, chem_b = _collect_rcsb_section_bodies(
        pmid,
        pmid_to_rcsb,
        enrich_uniprot=enrich_uniprot,
        enrich_chemical=enrich_chemical,
    )

    sections: List[str] = ["[Abstract]\n" + text.strip()]
    if struct_b:
        sections.append("[Structure info]\n" + struct_b)
    if eu and uni_b:
        sections.append("[UniProt]\n" + uni_b)
    if ec and chem_b:
        sections.append("[Chemical components]\n" + chem_b)

    if len(sections) == 1:
        return text

    combined = "\n\n".join(sections)
    # print(f"Combined length: {len(combined)}")
    if len(combined) > max_chars:
        
        combined = combined[:max_chars]
    return combined


def build_splits(
    docs: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 20,
):
    """
    Split documents into chunks using the same RecursiveCharacterTextSplitter
    configuration as the notebook.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def build_vectorstore(splits, persist: bool = True):
    """
    Create a Chroma vectorstore from document chunks using HuggingFace BGE embeddings.
    """
    embedding_model = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if persist:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma.from_documents(
            splits,
            embedding_model,
            persist_directory=str(CHROMA_DIR),
        )
    else:
        vectorstore = Chroma.from_documents(splits, embedding_model)

    return vectorstore


def build_retriever(vectorstore, splits, k: int = TOP_K):
    """
    Recreate the hybrid (dense + BM25) EnsembleRetriever from the notebook.
    """
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = k

    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.7, 0.3],  # 70% dense similarity + 30% lexical overlap
    )
    return retriever


_reranker: CrossEncoder | None = None


def build_reranker(model_name: str | None = None) -> CrossEncoder:
    """Load the cross-encoder reranker once (cached). Used by pipeline and notebook."""
    global _reranker
    if _reranker is None:
        name = model_name or RERANKER_MODEL
        _reranker = CrossEncoder(name)
    return _reranker


def rerank_entries(
    query: str,
    entries: List[Tuple[str, str, List[str]]],
    top_k: int,
    pmid_to_fulltext: dict[str, str] | None = None,
    model_name: str | None = None,
    pmid_to_rcsb: dict[str, List[dict]] | None = None,
    *,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
) -> List[Tuple[str, str, List[str]]]:
    """
    Rerank (pmid, full_abstract, pdb_ids) entries by relevance to query; return top_k.

    - If pmid_to_fulltext is provided, the reranker sees full text when available,
      otherwise the abstract.
    - If pmid_to_rcsb is provided, each document text (abstract or full text) is
      enriched with RCSB metadata (title, keywords, and optionally UniProt /
      chemical components; see enrich_uniprot / enrich_chemical and config defaults).
    """
    if not entries:
        return []

    reranker = build_reranker(model_name)

    texts: List[str] = []

    new_entries = []
    for pmid, abstract, _pdb_ids in entries:
        base_text = abstract
        if pmid_to_fulltext:
            full = pmid_to_fulltext.get(pmid)
            if full:
                base_text = full
        enriched = enrich_text_with_rcsb(
            pmid,
            base_text,
            pmid_to_rcsb,
            enrich_uniprot=enrich_uniprot,
            enrich_chemical=enrich_chemical,
        )
        # print(enriched)
        # print()
        texts.append(enriched)
        new_entries.append((pmid, base_text, _pdb_ids))
        # new_entries.append((pmid, enriched, _pdb_ids))

    pairs = [(query, text) for text in texts]
    reranker.max_seq_length = 4096#1024
    scores = reranker.predict(pairs)
    # scores = reranker.predict(pairs, max_length=1024)
    indexed = list(zip(scores, new_entries))#entries))
    indexed.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in indexed[:top_k]]


def build_llm(
    model_name: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = LLM_BATCH_SIZE,
):
    """
    Build a HuggingFacePipeline LLM using the Qwen model, following the notebook.
    """
    name = model_name or LLM_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe, batch_size=batch_size)


def _format_docs(docs):
    """Join document page_content for use as context."""
    return "\n\n".join(doc.page_content for doc in docs)


def _extract_final_answer(raw: str) -> str:
    """
    Post-process raw LLM output to return only the substantive answer.
    Strips whitespace and removes common prefixes like 'Answer:' or 'Assistant:'.
    """
    text = (raw or "").strip()
    if not text:
        return "No answer generated."

    lower = text.lower()
    labels = ["answer:", "assistant:", "reason:"]
    best_idx = -1
    best_label = ""
    for label in labels:
        idx = lower.rfind(label)
        if idx > best_idx:
            best_idx = idx
            best_label = label

    if best_idx != -1:
        text = text[best_idx + len(best_label) :].lstrip()

    return text or "No answer generated."


def _clean_reason(raw: str) -> str:
    """
    Remove prompt content from the reason chain output so the displayed
    reason contains only the model's answer (no echoed question, abstract, or instructions).
    """
    text = _extract_final_answer(raw or "")
    if not text or text == "No answer generated.":
        return text
    # If the model emits multiple "Reason:" sections, keep the last one.
    reason_sections = re.split(r"(?:^|\n)\s*reason:\s*", text, flags=re.IGNORECASE)
    if len(reason_sections) > 1:
        text = reason_sections[-1].strip()
    else:
        text = re.sub(r"^\s*reason:\s*", "", text, flags=re.IGNORECASE)
    # Remove markdown emphasis artifacts often emitted by instruction-tuned models.
    text = text.replace("**", "")
    lines = text.split("\n")
    kept: List[str] = []
    skip_prefixes = (
        "you are an experienced biomedical analyst",
        "you are a strict relevance analyst",
        "the user asked:",
        "user question:",
        "question:",
        "article abstract:",
        "article abstract and protein structure info:",
        "information of protein structure and related literature:",
        "task:",
        "rules:",
        "why is this article relevant to the question?",
        "answer in 1-2 sentences.",
        "output format:",
        "decision:",
        "output:",
    )
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        lower = stripped.lower()
        if any(lower.startswith(prefix) for prefix in skip_prefixes):
            continue
        # Drop common runaway instruction echoes.
        if (
            lower.startswith("end of ")
            or lower.startswith("do not ")
            or lower.startswith("no further ")
            or lower.startswith("thank you")
            or lower.startswith("- ")
        ):
            continue
        # Skip line that is only the instruction
        if lower in ("why is this article relevant to the question? answer in 1-2 sentences.",):
            continue
        kept.append(stripped)
    result = "\n".join(kept).strip()
    # Keep only the first 1-2 sentences for consistency with prompt constraints.
    if result:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", result) if s.strip()]
        if sents:
            result = " ".join(sents[:2]).strip()
    return result or "No reason generated."


def build_rag_chain(retriever, llm):
    """
    Construct the RAG chain equivalent to the notebook:
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
    """
    prompt = hub.pull("rlm/rag-prompt")
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_answer_only_chain(llm):
    """
    Chain that takes pre-filled context and question (no retriever).
    Used so we can retrieve once, then get both answer and PubMed IDs.
    """
    prompt = hub.pull("rlm/rag-prompt")
    return prompt | llm | StrOutputParser()

'''
    prompt = PromptTemplate.from_template(
         "You are an experienced biomedical analyst for biomedical retrieval.\n\n"
        "User question:\n{question}\n\n"
        "Information of protein structure and related literature:\n{info}\n\n"
        "Task: Analyzing the information of protein structure and related literature, whether these information could answers the user question or not? And why? The output should clearly indicate whether the info could answer the question or not using exactly 2-3 concise sentences.\n\n"
        "Rules: Do NOT output markdown, bullets, JSON, or extra sections. Keep the reason factual, direct. Do not offer further help."
    )
'''
def build_reason_chain(llm):
    """
    Chain that takes question and a single abstract and returns a brief
    reason why the article is relevant to the question.
    """
    prompt = PromptTemplate.from_template(
         "You are an experienced biomedical analyst for biomedical retrieval.\n\n"
        "User question:\n{question}\n\n"
        "Information of protein structure and related literature:\n{info}\n\n"
        "Task: Analyzing the information of protein structure and related literature, whether these information could be helpful to answer the user question or not? And why? The output should clearly indicate whether the info could be used to answer the question or not using exactly 2-3 concise sentences.\n\n"
        "Rules: Do NOT output markdown, bullets, JSON, or extra sections. Keep the reason factual, direct. Do not offer further help."
    )

    return prompt | llm | StrOutputParser()


def build_judge_chain(llm):
    """
    Chain that takes question, abstract, and reason and judges whether the article
    is relevant to the question or could answer it. Used for reranking (LLM-as-judge).
    """
    prompt = PromptTemplate.from_template(
        "A reason has been generated for relevance of the abstract and protein structure info to the question: {reason}\n\n"
        "Summarize the above reason to determine whether it indicates that the abstract and protein structure info is relevant to the question or not?"
        "Based on the summary of the reason, assign a relevance level:\n"
        "- 0 = Irrelevant to the question.\n"
        "- 1 = Relevant but only indirectly related. e.g., the reason indicates that the abstract and protein structure info could be adapted to answer the question.\n"
        "- 2 = Directly relevant to answering the question.\n\n"
        "In the answer, first, write exactly 1-2 concise sentences that starts with either "
        "'The abstract is irrelevant to the question because ...', "
        "'The abstract is somewhat related to the question because ...', or "
        "'The abstract is directly relevant to the question because ...'.\n"
        "Then, on a new line, output exactly one of 'SCORE: 0', 'SCORE: 1', or 'SCORE: 2' that matches your judgement."
    )


    return prompt | llm | StrOutputParser()


def build_query_rewrite_chain(llm):
    """
    Rewrite user queries for protein structure / PubMed–PDB retrieval: fix typos,
    complete fragmentary phrasing, and disambiguate when possible without adding
    constraints not implied by the original text.
    """
    prompt = PromptTemplate.from_template(
        "You rewrite user queries to improve the clarity, completeness, and remove ambiguity and typos in the query if needed. \n\n"
        "User query:\n{question}\n\n"
        "Instructions:\n"
        "- If needed, fix spelling and obvious typos.\n"
        "- If needed, turn incomplete or telegraphic text into one clear English question without changing the user’s scientific intent.\n"
        "- If the query is ambiguous, choose the most likely reading for structure/literature search; "
        "do not invent new biological constraints not implied above.\n"
        "Output must be a rewritten question (one short paragraph or single line). "
        "No preamble, labels, or explanation."
    )
    return prompt | llm | StrOutputParser()


def rewrite_query(chain: Any, raw_question: str) -> str:
    """Run the rewrite chain; post-process and fall back to the original on empty output."""
    raw = (raw_question or "").strip()
    if not raw:
        return ""
    raw_out = chain.invoke({"question": raw})
    if not isinstance(raw_out, str):
        raw_out = str(raw_out or "")
    text = _extract_final_answer(raw_out).strip()
    text = text.split("No preamble, labels, or explanation.")[-1]
    text = text.split('?')[0]+"?" if "?" in text else text
    return text if text else raw


def _prepare_query_for_retrieval(
    question: str,
    query_rewrite_chain: Any | None,
    skip_query_rewrite: bool = False,
) -> str:
    """Strip input; optionally LLM-rewrite when QUERY_REWRITE_ENABLED and chain is set."""
    raw = (question or "").strip()
    if not raw:
        return ""
    if skip_query_rewrite or not QUERY_REWRITE_ENABLED or query_rewrite_chain is None:
        return raw
    rewritten = rewrite_query(query_rewrite_chain, raw)
    return rewritten if rewritten else raw


def _parse_judge_score(raw: str) -> int:
    """
    Parse LLM judge output into an integer relevance score in {0, 1, 2}.
    Looks for 'SCORE: 0/1/2' or similar, with fallbacks from text cues.
    """
    text = (raw or "").strip()
    if not text:
        return 0
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("score"):
            for sep in (":", "=", "-"):
                if sep in lower:
                    _, after = lower.split(sep, 1)
                    parts = after.strip().split()
                    if parts and parts[0] in {"0", "1", "2"}:
                        return int(parts[0])
        if stripped in {"0", "1", "2"}:
            return int(stripped)
    lower_all = text.lower()
    if "directly relevant" in lower_all or "highly relevant" in lower_all:
        return 2
    if "somewhat related" in lower_all or "partially relevant" in lower_all:
        return 1
    if "not relevant" in lower_all or "irrelevant" in lower_all:
        return 0
    if "relevant" in lower_all or "yes" in lower_all:
        return 1
    return 0


def _parse_judge_output(raw: str) -> bool:
    """Parse LLM judge output to a boolean (True = relevant). Uses score >= 1."""
    return _parse_judge_score(raw) >= 1


def parse_judge_output(raw: str) -> bool:
    """Public alias for _parse_judge_output (e.g. for evaluation notebooks)."""
    return _parse_judge_output(raw)


def _judge_display_label(raw: str) -> str:
    """Normalize judge output for display: 'Irrelevant', 'Related', or 'Directly relevant'."""
    score = _parse_judge_score(raw)
    if score >= 2:
        return "Directly relevant"
    if score == 1:
        return "Related"
    return "Irrelevant"


def _format_judge_for_display(raw: str) -> str:
    """Return the full judge sentence for display (raw, lightly cleaned). Keeps the 'because ...' form."""
    raw = raw.split("matches your judgement.")[-1] if "matches your judgement." in raw else raw
    text = (raw or "").strip()
    return text if text else "No judgement generated."


def _chain_batch_in_chunks(chain: Any, inputs: List[dict], chunk_size: int) -> List[Any]:
    """Run chain.batch in slices so peak GPU memory stays bounded (chunk_size from LLM_BATCH_SIZE)."""
    if not inputs:
        return []
    size = max(1, chunk_size)
    out: List[Any] = []
    for i in range(0, len(inputs), size):
        out.extend(chain.batch(inputs[i : i + size]))
    return out


def _batch_reason_entries(
    cleaned_question: str,
    raw_entries: List[Tuple[str, str, List[str]]],
    reason_chain: Any,
    chunk_size: int,
) -> List[Tuple[str, str, List[str], str]]:
    """Run reason chain on reranked (pmid, info, pdb_ids) tuples."""
    reason_inputs = [
        {"question": cleaned_question, "info": info_text}
        for _, info_text, _ in raw_entries
    ]
    raw_reasons = _chain_batch_in_chunks(reason_chain, reason_inputs, chunk_size)
    entries_with_reason: List[Tuple[str, str, List[str], str]] = []
    for (pmid, info_text, pdb_ids), raw_reason in zip(raw_entries, raw_reasons, strict=False):
        raw_reason = raw_reason.split("Do not offer further help.")[1] if "Do not offer further help." in raw_reason else raw_reason
        reason = raw_reason.split("Output:")[1] if "Output:" in raw_reason else raw_reason
        # raw_reason = raw_reason.split("Reason:")[1] if "Reason:" in raw_reason else raw_reason
        # reason = _clean_reason(raw_reason if isinstance(raw_reason, str) else str(raw_reason or ""))
        entries_with_reason.append((pmid, info_text, pdb_ids, reason))
    return entries_with_reason


def _batch_judge_and_sort(
    cleaned_question: str,
    entries_with_reason: List[Tuple[str, str, List[str], str]],
    judge_chain: Any,
    chunk_size: int,
) -> List[EntryTuple]:
    """Run judge chain and stable-sort by descending score."""
    judge_inputs = [
        {
            "question": cleaned_question,
            "info": info_text,
            "reason": reason,
        }
        for _, info_text, _, reason in entries_with_reason
    ]
    raw_judgements = _chain_batch_in_chunks(judge_chain, judge_inputs, chunk_size)
    all_with_judgement: List[EntryTuple] = []
    for entry, raw_judge in zip(entries_with_reason, raw_judgements, strict=False):
        pmid, info_text, pdb_ids, reason = entry
        rj = raw_judge if isinstance(raw_judge, str) else str(raw_judge or "")
        judgement = _format_judge_for_display(rj)
        judgement = judgement.split("SCORE")[0]
        score = _parse_judge_score(rj)
        is_relevant = score >= 1
        all_with_judgement.append(
            (pmid, info_text, pdb_ids, reason, judgement, score, is_relevant)
        )
    all_with_judgement.sort(key=lambda e: -e[5])
    return all_with_judgement


def _entries_after_reason_judge(
    cleaned_question: str,
    raw_entries: List[Tuple[str, str, List[str]]],
    reason_chain: Any,
    judge_chain: Any,
    chunk_size: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[EntryTuple]:
    """
    Given reranked (pmid, abstract text for prompts, pdb_ids), run reason → judge → sort by score.
    Matches UI post-rerank behavior (stable sort by descending judge score).
    """
    n = len(raw_entries)
    if progress_callback:
        progress_callback(
            0.3,
            f"Generating relevance for {n} reranked articles in one GPU batch...",
        )
    entries_with_reason = _batch_reason_entries(
        cleaned_question, raw_entries, reason_chain, chunk_size
    )
    if progress_callback:
        progress_callback(0.85, f"Judging relevance for {n} articles in one GPU batch...")
    return _batch_judge_and_sort(
        cleaned_question, entries_with_reason, judge_chain, chunk_size
    )


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
    skip_query_rewrite: bool = False,
    *,
    enrich_uniprot: bool | None = None,
    enrich_chemical: bool | None = None,
) -> Tuple[List[str], List[EntryTuple]]:
    """
    Retrieve → cross-encoder rerank → reason → judge → sort by judge score.

    Returns (ordered_pmids, full entries) for evaluation or UI. Reranker enrichment follows
    ``rerank_entries`` (RCSB snippet in cross-encoder input when ``pmid_to_rcsb`` is set);
    reason/judge see the abstract text field returned from reranking (same as UI).
    Optional ``enrich_uniprot`` / ``enrich_chemical`` override config for reranker text.

    When ``RAD_RAG_QUERY_REWRITE`` is set and ``query_rewrite_chain`` is provided (unless
    ``skip_query_rewrite``), the query is rewritten once before retrieval and reranking.
    """
    cleaned_question = _prepare_query_for_retrieval(
        question, query_rewrite_chain, skip_query_rewrite)
    if not cleaned_question:
        return [], []

    if progress_callback:
        progress_callback(0.1, "Retrieving relevant documents...")
    docs_retrieved = retriever.get_relevant_documents(cleaned_question)

    raw_entries: List[Tuple[str, str, List[str]]] = []
    for doc in docs_retrieved:
        pmid = str(doc.metadata.get("PubMed_ID", "?"))
        full_abstract = pmid_to_abstract.get(pmid, doc.page_content)
        pdb_ids = pmid_to_pdb.get(pmid, [])
        raw_entries.append((pmid, full_abstract, pdb_ids))

    if progress_callback:
        progress_callback(0.15, "Reranking retrieved articles...")
    raw_entries = rerank_entries(
        cleaned_question,
        raw_entries,
        top_k=top_k,
        pmid_to_fulltext=pmid_to_fulltext,
        pmid_to_rcsb=pmid_to_rcsb,
        enrich_uniprot=enrich_uniprot,
        enrich_chemical=enrich_chemical,
    )

    chunk = llm_batch_size if llm_batch_size is not None else LLM_BATCH_SIZE
    entries = _entries_after_reason_judge(
        cleaned_question,
        raw_entries,
        reason_chain,
        judge_chain,
        chunk,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback(1.0, "Done.")
    return [e[0] for e in entries], entries


EntryTuple = Tuple[str, str, List[str], str, str, int, bool]  # (pmid, full_abstract, pdb_ids, reason, judgement, score, is_relevant)
AnswerAndEntriesSig = Callable[
    [str, Optional[Callable[[float, str], None]]], Tuple[str, List[EntryTuple]]
]
StreamingYield = Tuple[str, str] | Tuple[str, str, List[EntryTuple]]  # ("status", msg) or ("result", answer, entries)
AnswerAndEntriesStreamSig = Callable[[str], Generator[StreamingYield, None, None]]
_answer_fn: AnswerAndEntriesSig | None = None
_streaming_fn: AnswerAndEntriesStreamSig | None = None


def initialize_pipeline() -> Tuple[AnswerAndEntriesSig, AnswerAndEntriesStreamSig]:
    """
    Initialize the full RAG pipeline once and return (answer_and_entries, answer_and_entries_streaming).
    - answer_and_entries(question) -> (answer, entries) for entries (pmid, full_abstract, pdb_ids, reason, judgement, is_relevant).
    - answer_and_entries_streaming(question) yields ("status", message) then ("result", answer, entries).
    """
    global _answer_fn, _streaming_fn
    if _answer_fn is not None and _streaming_fn is not None:
        return _answer_fn, _streaming_fn

    print("[RAG] Loading documents from JSON...")
    docs = load_documents()

    print("[RAG] Loading PMID -> full abstract, PMID -> PDB IDs, and RCSB metadata lookups...")
    pmid_to_abstract = load_pmid_to_abstract()
    pmid_to_pdb = load_pmid_to_pdb_ids()
    pmid_to_rcsb = load_pmid_to_rcsb_metadata()

    print(
        "[RAG] Reranker enrichment flags: "
        f"UniProt={ENRICH_RERANK_UNIPROT}, chemical={ENRICH_RERANK_CHEMICAL} "
        f"(set RAD_RAG_ENRICH_UNIPROT / RAD_RAG_ENRICH_CHEMICAL to 0 or 1 to compare)"
    )

    # [ablation]if use rcsb structure info, uncomment the following code
    # for d in docs:
    #     pmid = str(d.metadata.get("PubMed_ID", ""))
    #     d.page_content = enrich_text_with_rcsb(pmid, d.page_content, pmid_to_rcsb)

    print(f"[RAG] Loaded {len(docs)} documents. Building splits...")
    splits = build_splits(docs)

    print(f"[RAG] Built {len(splits)} chunks. Creating vectorstore...")
    vectorstore = build_vectorstore(splits, persist=True)

    print("[RAG] Creating hybrid retriever (dense + BM25)...")
    retriever = build_retriever(vectorstore, splits, k=RETRIEVAL_DISPLAY_K)

    print(f"[RAG] Loading reranker: {RERANKER_MODEL}...")
    build_reranker()

    print(f"[RAG] Loading LLM: {LLM_MODEL_NAME}...")
    llm = build_llm()

    print("[RAG] Building reason chain (per-article relevance)...")
    reason_chain = build_reason_chain(llm)
    print("[RAG] Building judge chain (reranking, LLM-as-judge)...")
    judge_chain = build_judge_chain(llm)

    query_rewrite_chain: Any | None = None
    if QUERY_REWRITE_ENABLED:
        print("[RAG] Building query rewrite chain (pre-retrieval)...")
        query_rewrite_chain = build_query_rewrite_chain(llm)

    def answer_and_entries(
        question: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
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
        )
        n = len(all_with_judgement)
        answer = (
            f"Retrieved articles; showing the top {n} sorted by relevance score (0-2), with ties in reranker order. "
            "Below: each article's PMID, PDB IDs, reason, judgement, score, and full abstract (relevant entries highlighted)."
        )
        return answer, all_with_judgement

    def answer_and_entries_streaming(
        question: str,
    ) -> Generator[StreamingYield, None, None]:
        """Yields ('status', message) then ('result', answer, entries) for live UI updates."""
        if not question.strip():
            yield ("result", "Please provide a non-empty question.", [])
            return
        if QUERY_REWRITE_ENABLED and query_rewrite_chain is not None:
            yield ("status", "Rewriting query...")
        cleaned_question = _prepare_query_for_retrieval(
            question, query_rewrite_chain, skip_query_rewrite=False)
        if not cleaned_question:
            yield ("result", "Please provide a non-empty question.", [])
            return

        yield ("status", "Retrieving relevant documents...")
        docs_retrieved = retriever.get_relevant_documents(cleaned_question)

        raw_entries: List[Tuple[str, str, List[str]]] = []
        for doc in docs_retrieved:
            pmid = str(doc.metadata.get("PubMed_ID", "?"))
            full_abstract = pmid_to_abstract.get(pmid, doc.page_content)
            pdb_ids = pmid_to_pdb.get(pmid, [])
            raw_entries.append((pmid, full_abstract, pdb_ids))

        yield ("status", "Reranking retrieved articles...")
        raw_entries = rerank_entries(
            cleaned_question,
            raw_entries,
            top_k=RERANK_TOP_K,
            pmid_to_fulltext=None,
            pmid_to_rcsb=pmid_to_rcsb,
        )

        n = len(raw_entries)
        yield (
            "status",
            f"Generating relevance for {n} reranked articles in one GPU batch...",
        )
        entries_with_reason = _batch_reason_entries(
            cleaned_question, raw_entries, reason_chain, LLM_BATCH_SIZE
        )
        yield ("status", f"Judging relevance for {n} articles in one GPU batch...")
        all_with_judgement = _batch_judge_and_sort(
            cleaned_question, entries_with_reason, judge_chain, LLM_BATCH_SIZE
        )

        yield ("status", "Done.")
        answer = (
            f"Retrieved articles; showing the top {n} sorted by relevance score (0-2), with ties in reranker order. "
            "Below: each article's PMID, PDB IDs, reason, judgement, score, and full abstract (relevant entries highlighted)."
        )
        yield ("result", answer, all_with_judgement)

    _answer_fn = answer_and_entries
    _streaming_fn = answer_and_entries_streaming
    print("[RAG] Pipeline initialization complete.")
    return _answer_fn, _streaming_fn
