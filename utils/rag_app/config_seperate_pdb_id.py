from __future__ import annotations

import os
from pathlib import Path


# Base project directory (RAD_protein root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Default data file used in the notebook
DEFAULT_DATA_PATH: Path = PROJECT_ROOT / "demo_pubmed.json"

# RCSB data for PMID -> PDB ID mapping (from 1. Single_abs_question_generation_2026.ipynb)
DEFAULT_RCSB_DATA_PATH: Path = PROJECT_ROOT / "full_rcsb_data_04132026.json"

# Abbreviation dictionary used to expand unambiguous abbreviations in user questions
DEFAULT_ABBR_DICT_PATH: Path = PROJECT_ROOT / "Official_annotation" / "abbr_dict_all_abs.csv"

# Where to store the Chroma persistent database by default
DEFAULT_CHROMA_DIR: Path = PROJECT_ROOT / "test" / "chroma_db"


# Public configuration values (can be overridden via environment variables)
DATA_PATH: Path = Path(
    os.getenv("RAD_RAG_DATA_PATH", str(DEFAULT_DATA_PATH))
)

CHROMA_DIR: Path = Path(
    os.getenv("RAD_RAG_CHROMA_DIR", str(DEFAULT_CHROMA_DIR))
)

RCSB_DATA_PATH: Path = Path(
    os.getenv("RAD_RAG_RCSB_DATA_PATH", str(DEFAULT_RCSB_DATA_PATH))
)

ABBR_DICT_PATH: Path = Path(
    os.getenv("RAD_RAG_ABBR_DICT_PATH", str(DEFAULT_ABBR_DICT_PATH))
)

# When true, expand unambiguous abbreviations in the user question to
# "full name (ABBR)" before retrieval.
ABBR_EXPANSION_ENABLED: bool = bool(int(os.getenv("RAD_RAG_ABBR_EXPANSION", "1")))

# Model names mirror those used in the notebook
LLM_MODEL_NAME: str = os.getenv(
    "RAD_RAG_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)

# Batch size for Hugging Face text-generation calls. Set this >= RERANK_TOP_K
# to send all reason/judgement prompts for one query to the GPU together.
LLM_BATCH_SIZE: int = int(os.getenv("RAD_RAG_LLM_BATCH_SIZE", "16"))

EMBEDDING_MODEL_NAME: str = os.getenv(
    "RAD_RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
)

# Top-k documents for retrieval (legacy / backward compatibility)
TOP_K: int = int(os.getenv("RAD_RAG_TOP_K", "1"))

# Number of documents to retrieve and show in the UI (e.g. top-k PubMed IDs)
RETRIEVAL_DISPLAY_K: int = int(os.getenv("RAD_RAG_RETRIEVAL_K", "10"))

# Reranker: top-k after reranking to pass to reason/judge (UI pipeline only)
RERANK_TOP_K: int = int(os.getenv("RAD_RAG_RERANK_TOP_K", "10"))

# Reranker model (cross-encoder, e.g. BAAI/bge-reranker-base)
RERANKER_MODEL: str = os.getenv("RAD_RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")

# Tunnel mode for public URL access.
# "cloudflared" (default) = Cloudflare Quick Tunnel (reliable, no account needed)
# "gradio"               = Gradio share tunnel (legacy, less reliable)
# "none"                 = No public URL (local/LAN only)
TUNNEL_MODE: str = os.getenv("RAD_RAG_TUNNEL", "cloudflared").lower()

# When true, run an LLM query rewrite (typos, incomplete, ambiguous) before retrieval.
QUERY_REWRITE_ENABLED: bool = bool(int(os.getenv("RAD_RAG_QUERY_REWRITE", "0")))

# When true, run query decomposition before retrieval/reranking.
QUERY_DECOMPOSITION_ENABLED: bool = bool(
    int(os.getenv("RAD_RAG_QUERY_DECOMPOSITION", "0"))
)

# RRF constant for aggregating per-subquery ranked PMID lists.
QUERY_DECOMPOSITION_RRF_K: int = int(
    os.getenv("RAD_RAG_QUERY_DECOMPOSITION_RRF_K", "60")
)

# Reranker text enrichment (RCSB structure snippet appended to abstract/full text).
# Set each to 0 or 1 to compare ablations.
# default: 0 for no enrichment
ENRICH_RERANK_STRUCTURE: bool = bool(int(os.getenv("RAD_RAG_ENRICH_STRUCTURE", "0")))
ENRICH_RERANK_UNIPROT: bool = bool(int(os.getenv("RAD_RAG_ENRICH_UNIPROT", "0")))
ENRICH_RERANK_CHEMICAL: bool = bool(int(os.getenv("RAD_RAG_ENRICH_CHEMICAL", "0")))

# Indexing-time metadata concat for embedding input.
# When enabled, each embedded chunk can include selected RCSB sections.
# default: 0 for no concatenation
INDEX_CONCAT_METADATA: bool = bool(int(os.getenv("RAD_RAG_INDEX_CONCAT_METADATA", "0")))
INDEX_CONCAT_STRUCTURE: bool = bool(int(os.getenv("RAD_RAG_INDEX_CONCAT_STRUCTURE", "1")))
INDEX_CONCAT_UNIPROT: bool = bool(int(os.getenv("RAD_RAG_INDEX_CONCAT_UNIPROT", "0")))
INDEX_CONCAT_CHEMICAL: bool = bool(int(os.getenv("RAD_RAG_INDEX_CONCAT_CHEMICAL", "0")))

# Caps keep cross-encoder inputs reasonable (see pipeline reranker max_seq_length).
ENRICH_UNIPROT_MAX_CHARS: int = int(os.getenv("RAD_RAG_ENRICH_UNIPROT_MAX_CHARS", "800"))
ENRICH_CHEM_MAX_NAMES: int = int(os.getenv("RAD_RAG_ENRICH_CHEM_MAX_NAMES", "35"))

# Per-PDB rerank aggregation policy when collapsing multiple (PMID, PDB) scores.
# Accepted values: "max" (default), "mean".
RERANK_PDB_AGGREGATION: str = os.getenv("RAD_RAG_PDB_AGGREGATION", "max").strip().lower()
if RERANK_PDB_AGGREGATION not in {"max", "mean"}:
    RERANK_PDB_AGGREGATION = "max"

# If true, include abstract-only candidates for PDB IDs that lack RCSB metadata rows.
INCLUDE_MISSING_PDB_AS_ABSTRACT_ONLY: bool = bool(
    int(os.getenv("RAD_RAG_INCLUDE_MISSING_PDB_AS_ABSTRACT_ONLY", "0"))
)
