# RAD-PROTEIN

**RAD-PROTEIN: Multi-source Knowledge-enriched Retrieval-augmented Protein Structure Searching**

This repository contains code to:

1. generate natural-language protein structure search questions from PubMed abstracts linked to RCSB PDB entries;
2. build an abbreviation dictionary and abbreviation variants of the question set;
3. evaluate the RAD-PROTEIN retrieval pipeline (hybrid retrieval → cross-encoder reranking enriched with RCSB metadata) on single-abstract questions.

A small example dataset (`example_data/`, 5 samples per file) ships with the repo, so all three notebooks run end to end out of the box.

## Pipeline

```
RCSB PDB metadata  +  PubMed abstracts
              │
              ▼
[1] Question generation (Qwen2.5-7B-Instruct, 2-stage: generate 5 → select best)      GPU
              │
              ▼
    Human annotation  →  curated single-abstract question set
              │
              ▼
[2] Abbreviation dictionary + abbreviation variants of each question                  GPU
              │
              ▼
[3] Retrieval evaluation: retrieve-only vs. rerank (plain / RCSB-enriched)            GPU
    Recall@k, Precision@k, MRR, MAP, NDCG@k with 95% bootstrap CIs
```

## Repository structure

```
RAD_PROTEIN/
├── 1. Preprocessing_Single_abs_question_generation_2026.ipynb   # question generation
├── 2. abbreviation_dataset_dictionary.ipynb                     # abbreviation dictionary + variants
├── 3. Evaluation_single_abs_question.ipynb                      # retrieval evaluation
├── get_all_basic_rcsb_data.py        # downloads full RCSB PDB metadata (full_rcsb_data.json)
├── requirements.txt
├── utils/rag_app/
│   ├── config.py                     # base config (env-var overridable)
│   ├── config_seperate_pdb_id.py     # config for per-PDB reranking (used in notebook 3)
│   ├── pipeline.py                   # loading, chunking, Chroma + BM25 retrieval, reranker, LLM chains
│   ├── pipeline_seperate_pdb_id.py   # per-(PMID, PDB) reranking with RCSB metadata enrichment
│   └── abbreviation.py               # abbreviation dictionary loading + query expansion
├── example_data/                     # 5-sample inputs in the same format as the full data
│   ├── demo_pubmed.json.gz           # PubMed abstracts: [{"PubMed_ID", "Abstract", ...}] (unzip before use)
│   ├── full_rcsb_data.json           # RCSB entries: {"data": {"entries": [...]}}
│   └── Official_annotation/
│       ├── single_expand_example.csv                     # questions (question, pmcid, ...)
│       ├── single_expand_example_with_abbr_variants.csv  # + with/without-abbreviation columns
│       ├── abbr_dict_example.csv                         # PubMed_ID, abbreviation, full_name
│       └── Official_annotation_R{1,2,3}/used_info_*.json # categories/PMIDs/PDB IDs already used
└── outputs/                          # created by the notebooks (git-ignored)
```

### About the example data

- The example questions are Qwen-generated questions that are **not** part of the curated evaluation benchmark. The annotation columns (`Round`, `Name`, `Rate_quality`, `Relevance`, `Complexity`, `Confidence`, `Other_comments`) are kept so the file format matches, but they are left empty.
- `demo_pubmed.json.gz` and `full_rcsb_data.json` contain only the 5 abstracts and RCSB entries behind those questions. Retrieval metrics on the example data are therefore trivially high: the example is a smoke test, not a benchmark.
- `used_info_*.json` lists a few previously used categories, PMIDs and PDB IDs, using the same keys as the real files.

## Setup

### 1. Environment

```bash
python -m venv rad_env
source rad_env/bin/activate
pip install -r requirements.txt
```

All three notebooks need a CUDA GPU. Qwen2.5-7B-Instruct in bf16 needs roughly 16 GB of GPU memory, and notebook 3 also loads BGE-M3 and bge-reranker-large. Models are downloaded from Hugging Face on first use. Notebook 1 places the model on `cuda:1` (`device_map={"": 1}` in cell 2); change this to match your machine.

### 2. Data

The notebooks read from `example_data/` by default. The PubMed corpus is stored gzipped, so unzip it back to JSON **before running any notebook**:

```bash
gunzip -k example_data/demo_pubmed.json.gz   # → example_data/demo_pubmed.json (-k keeps the .gz)
```

To run on the full data:

1. Create a data folder with the same layout as `example_data/`.
2. Generate the RCSB metadata (~2.3 GB):
   ```bash
   python get_all_basic_rcsb_data.py      # → full_rcsb_data.json
   ```
3. Add the PubMed corpus as an unzipped `demo_pubmed.json`, your question CSVs and the abbreviation dictionary.
4. In each notebook, set `DATA_DIR` in the path cell to your folder, and update the example file names (`single_expand_example*.csv`, `abbr_dict_example.csv`).

## Notebooks

Run Jupyter from the repository root (`jupyter notebook`). Each notebook defines `PROJECT_ROOT`, `DATA_DIR`, `ANNOTATION_DIR` and `OUTPUT_DIR` in its first path cell. Results are written to `outputs/<step>/`, so the input files are never overwritten.

### `1. Preprocessing_Single_abs_question_generation_2026.ipynb`

Generates search-style questions ("Can you find structures of …") from individual PubMed abstracts.

- Loads `full_rcsb_data.json` and builds PDB ID ↔ PubMed ID ↔ abstract lookups.
- Samples source-organism categories (`selected = 'single'`: categories with 1–30 entries; `'multiple'`: more than 30 entries), skipping categories and PMIDs listed in `Official_annotation_R*/used_info_*.json`.
- Stage 1: Qwen2.5-7B-Instruct writes five candidate questions per abstract. Stage 2: the same model selects the single most useful one.
- Parses the output into `{Question, Reference, Pubmed_IDs, RCSB_PDB_IDs}` records.
- Output: `outputs/question_generation/qwen_<n>_single_abstract_*.json` and an updated `used_info_single.json`.

Settings: `model_type` (`qwen` / `llama` / `gemini`) in cell 2, and `selected`.

### `2. abbreviation_dataset_dictionary.ipynb`

- **Question variants:** detects `full name (ABBR)` and `ABBR (full name)` patterns in each question and adds `with_abbreviation` / `without_abbreviation` columns. Output: `outputs/abbreviation/<name>_with_abbr_variants.csv`.
- **Abbreviation dictionary:** extracts abbreviation–definition pairs from every abstract in `demo_pubmed.json`, using regex and Qwen2.5-7B-Instruct. Output: `outputs/abbreviation/abbr_dict_all_abs.csv`.

### `3. Evaluation_single_abs_question.ipynb`

Evaluates retrieval on the single-abstract questions. The ground truth for each question is its source PMID.

Retrieval: chunks (size 450, overlap 20) from `demo_pubmed.json` are indexed with BGE-M3 in Chroma and combined with BM25 in a hybrid ensemble retriever (top 100). Conditions compared:

| # | Condition | Reranker input |
|---|-----------|----------------|
| 1 | Retrieve-only | none |
| 2 | Rerank (plain) | abstract |
| 3 | Rerank (enriched, structure) | abstract + RCSB title/keywords |
| 4 | Rerank (enriched, structure + UniProt) | + UniProt function text |
| 5 | Rerank (enriched, structure + chemical) | + chemical component names |
| 6 | Rerank (enriched, structure + UniProt + chemical) | all of the above |

Metrics: Recall@k, Precision@k and NDCG@k (k ∈ {1, 5, 10, 20, 50}), plus MRR and MAP. They are reported as mean ± std and with 95% bootstrap CIs (10,000 resamples), and saved to `outputs/evaluation/`.

Flags in the first cell:

| Flag | Effect |
|------|--------|
| `concat_chunk` | append RCSB metadata to chunks before embedding |
| `abbr_rewrite_mode` | use abbreviated questions and expand them with the abbreviation dictionary before retrieval |
| `abbr_baseline` | use abbreviated questions without expansion |
| `dense_only` / `sparse_only` | Chroma-only or BM25-only retrieval baselines |

Note: the Chroma index in `chroma_db_eval/` is deleted and rebuilt on every run.

## Configuration

`utils/rag_app/config_seperate_pdb_id.py` reads the following environment variables. Notebook 3 sets them in its path cell.

| Variable | Set in notebook 3 to | Meaning |
|----------|----------------------|---------|
| `RAD_RAG_DATA_PATH` | `example_data/demo_pubmed.json` | PubMed corpus |
| `RAD_RAG_RCSB_DATA_PATH` | `example_data/full_rcsb_data.json` | RCSB metadata |
| `RAD_RAG_ABBR_DICT_PATH` | `example_data/Official_annotation/abbr_dict_example.csv` | abbreviation dictionary |
| `RAD_RAG_CHROMA_DIR` | `chroma_db_eval/` | Chroma persistence directory |
| `RAD_RAG_EMBED_MODEL` | `BAAI/bge-m3` | embedding model |
| `RAD_RAG_RERANKER_MODEL` | `BAAI/bge-reranker-large` | cross-encoder reranker |
| `RAD_RAG_LLM_MODEL` | (default) `Qwen/Qwen2.5-7B-Instruct` | LLM for rewrite / reason / judge chains |
| `RAD_RAG_PDB_AGGREGATION` | `max` | how per-PDB rerank scores are collapsed per PMID (`max` / `mean`) |

## Citation

Citation information will be added when the paper is published.
