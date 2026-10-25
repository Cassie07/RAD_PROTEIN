from __future__ import annotations

import os
from pathlib import Path


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    raw = _first_env(*names, default=str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    raw = _first_env(*names, default=str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(*names: str, default: bool) -> bool:
    raw = _first_env(*names, default=("1" if default else "0")).lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


# Base project directory (RAD_protein root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Default data file used in the notebook
DEFAULT_DATA_PATH: Path = PROJECT_ROOT / "demo_pubmed.json"

# RCSB data for PMID -> PDB ID mapping (from 1. Single_abs_question_generation_2026.ipynb)
DEFAULT_RCSB_DATA_PATH: Path = PROJECT_ROOT / "full_rcsb_data_04132026.json"

# Where to store the Chroma persistent database by default
DEFAULT_CHROMA_DIR: Path = PROJECT_ROOT / "test" / "chroma_db"

# Where to store async batch JSON result files
DEFAULT_BATCH_RESULTS_DIR: Path = PROJECT_ROOT / "test" / "rag_batch_results"


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

BATCH_RESULTS_DIR: Path = Path(
    os.getenv("RAD_RAG_BATCH_RESULTS_DIR", str(DEFAULT_BATCH_RESULTS_DIR))
)

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

# Public URL readiness probing knobs (used by cloudflared mode).
TUNNEL_READY_TIMEOUT_SECONDS: float = _env_float(
    "RAD_RAG_TUNNEL_READY_TIMEOUT", default=180.0
)
TUNNEL_READY_POLL_SECONDS: float = _env_float(
    "RAD_RAG_TUNNEL_READY_POLL", default=2.0
)

# When true, run an LLM query rewrite (typos, incomplete, ambiguous) before retrieval.
QUERY_REWRITE_ENABLED: bool = bool(int(os.getenv("RAD_RAG_QUERY_REWRITE", "0")))

# Reranker text enrichment (RCSB structure snippet appended to abstract/full text).
# Set each to 0 or 1 to compare ablations: structure title/keywords always included when
# any RCSB row exists; these flags add UniProt function text and chemical component names.
# default: 0 for no enrichment
ENRICH_RERANK_UNIPROT: bool = bool(int(os.getenv("RAD_RAG_ENRICH_UNIPROT", "0")))
ENRICH_RERANK_CHEMICAL: bool = bool(int(os.getenv("RAD_RAG_ENRICH_CHEMICAL", "0")))

# Caps keep cross-encoder inputs reasonable (see pipeline reranker max_seq_length).
ENRICH_UNIPROT_MAX_CHARS: int = int(os.getenv("RAD_RAG_ENRICH_UNIPROT_MAX_CHARS", "800"))
ENRICH_CHEM_MAX_NAMES: int = int(os.getenv("RAD_RAG_ENRICH_CHEM_MAX_NAMES", "35"))

# SMTP email settings for async batch completion notifications.
SMTP_HOST: str = _first_env(
    "RAD_RAG_SMTP_HOST",
    "SMTP_HOST",
    "SMTP_SERVER",
    "EMAIL_HOST",
    "MAIL_SERVER",
)
SMTP_PORT: int = _env_int(
    "RAD_RAG_SMTP_PORT",
    "SMTP_PORT",
    "EMAIL_PORT",
    "MAIL_PORT",
    default=587,
)
SMTP_USERNAME: str = _first_env(
    "RAD_RAG_SMTP_USERNAME",
    "SMTP_USERNAME",
    "EMAIL_HOST_USER",
    "MAIL_USERNAME",
)
SMTP_PASSWORD: str = _first_env(
    "RAD_RAG_SMTP_PASSWORD",
    "SMTP_PASSWORD",
    "EMAIL_HOST_PASSWORD",
    "MAIL_PASSWORD",
)
SMTP_FROM_EMAIL: str = _first_env(
    "RAD_RAG_SMTP_FROM_EMAIL",
    "SMTP_FROM_EMAIL",
    "EMAIL_FROM",
    "MAIL_FROM",
    default=SMTP_USERNAME,
)
SMTP_USE_TLS: bool = _env_bool(
    "RAD_RAG_SMTP_USE_TLS",
    "SMTP_USE_TLS",
    "EMAIL_USE_TLS",
    "MAIL_USE_TLS",
    default=True,
)
SMTP_USE_SSL: bool = _env_bool(
    "RAD_RAG_SMTP_USE_SSL",
    "SMTP_USE_SSL",
    "EMAIL_USE_SSL",
    "MAIL_USE_SSL",
    default=False,
)

EMAIL_ATTACH_JSON: bool = _env_bool(
    "RAD_RAG_EMAIL_ATTACH_JSON",
    "EMAIL_ATTACH_JSON",
    default=True,
)
EMAIL_MAX_ATTACH_MB: float = _env_float(
    "RAD_RAG_EMAIL_MAX_ATTACH_MB",
    "EMAIL_MAX_ATTACH_MB",
    default=20.0,
)

