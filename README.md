# UrduHaluBench

**Benchmarking hallucination, abstention and hallucination detectors in Urdu retrieval-augmented generation (RAG).**

Large language models used in RAG systems are supposed to answer *only* from the retrieved passage — and to say so when the passage does not contain the answer. Almost all tools and datasets for checking this are English-only. UrduHaluBench asks:

1. Do LLMs hallucinate more when the same RAG task is posed in **Urdu** instead of English?
2. Do they still **abstain** ("the passage does not contain the answer") when they should?
3. Do existing **hallucination detectors**, built and validated in English, still work on Urdu?

The benchmark is **parallel**: every item exists in English and Urdu with the same passage and question, so differences can be attributed to the language.

> **Status:** work in progress (MS research project, FAST-NUCES, 2026). Benchmark and model answers are released; human labels and detector results will follow.

---

## Benchmark

350 items, each in English and Urdu.

| Source | Answerable | Should abstain | Total |
|---|---|---|---|
| [UQA](https://github.com/sameearif/UQA) (Urdu SQuAD 2.0) + English SQuAD 2.0 | 175 | 75 — *unanswerable*: the passage is on topic but lacks the answer | 250 |
| [LEGAL-UQA](https://github.com/nlp-anonymous-researcher/LEGAL-UQA) (Constitution of Pakistan) | 70 | 30 — *wrong context*: an unrelated constitutional article is given | 100 |
| **Total** | **245** | **105 (30%)** | **350** |

- One question per passage; all 35 UQA topics covered; fixed random seeds (reproducible).
- **Wrong-context items** were built by swapping in a random other article; all were checked by hand and 2 that still allowed an answer were replaced.
- **Translation check:** a bilingual review of the 250 machine-translated UQA items removed 10 (4%) with serious errors or `<unk>` artifacts — details in [`docs/translation_check_v1.md`](docs/translation_check_v1.md).
- `benchmark_v1` is the main benchmark; `v2`–`v5` are alternative samples (different seeds) for robustness checks.

### Item format (`data/benchmark/benchmark_v1.csv`)
| Column | Description |
|---|---|
| `item_id` | `UHB-0001` … `UHB-0350` |
| `source` | `uqa` or `legal_uqa` |
| `condition` | `answerable`, `unanswerable`, `wrong_context` |
| `expected_behavior` | `answer` or `abstain` |
| `question_en` / `question_ur` | question |
| `context_en` / `context_ur` | passage given to the model |
| `gold_answer_en` / `gold_answer_ur` | reference answer(s), `\|\|`-separated; empty for abstain items |
| `topic`, `source_id` | original topic and id in the source dataset |

---

## Models

All models get the same instruction ([`scripts/prompts.py`](scripts/prompts.py)): answer only from the passage, otherwise reply with a fixed "not in the passage" sentence (English or Urdu).

| Model | Access | Decoding |
|---|---|---|
| Claude Haiku 5.5 (`claude-haiku-5-5`) | Anthropic API | model default (fixed temperature not supported) |
| Qwen3-8B (`Qwen/Qwen3-8B`) | open weights, 4-bit, Colab T4 | greedy, thinking mode off |
| Llama-3.1-8B-Instruct | open weights, 4-bit, Colab T4 | greedy |

Answers: `data/generations/<model>.csv` (700 per model = 350 items × 2 languages).

---

## Labels and metrics

Each answer is labelled against the **passage only** (not real-world truth):

| Label | Meaning |
|---|---|
| `F` | Faithful: answered, fully supported by the passage |
| `H` | Hallucinated: unsupported or contradicting claim (`INT` contradicts / `EXT` adds) |
| `CA` | Correct abstention: said "not in the passage" on an abstain item |
| `WA` | Wrong abstention: refused an answerable item |

Guides: [`docs/HOW_TO_ANNOTATE.md`](docs/HOW_TO_ANNOTATE.md) (short) and [`docs/annotation_guideline.md`](docs/annotation_guideline.md) (full rules and metric definitions).

Metrics ([`scripts/05_evaluate.py`](scripts/05_evaluate.py)), each with bootstrap 95% confidence intervals:
- **Generation:** hallucination rate, abstention recall, over-refusal rate — per model and language.
- **Detectors** (planned: HHEM, RAGAS faithfulness, LLM-as-judge): precision, recall, F1, AUROC.
- **English–Urdu gap:** ΔF1 / ΔAUROC with McNemar's exact test on paired items.

---

## Repository structure

```
UrduHaluBench/
├── scripts/
│   ├── 01_prepare_data.py        download UQA, SQuAD 2.0, LEGAL-UQA → data/
│   ├── 02_sample_benchmark.py    build benchmark_v1 … v5
│   ├── 03_generate.py            run a model over the benchmark (API / Hugging Face / dummy)
│   ├── prompts.py                the fixed English and Urdu prompts
│   ├── 04_annotation_sheets.py   build / check the annotation sheet
│   └── 05_evaluate.py            compute all metrics
├── notebooks/run_open_models.ipynb   Colab notebook for Qwen and Llama
├── data/
│   ├── benchmark/                benchmark_v1 … v5 (released)
│   ├── generations/              model answers (released)
│   └── annotations/              annotation sheet → final labelled dataset
├── docs/                         annotation guides, model-running guide, translation check
├── example.env                   template for API keys
└── requirements.txt
```
Source datasets are **not** stored in the repo (`.gitignore`); `01_prepare_data.py` downloads them into `raw/` and `data/`.

---

## Reproducing

Python 3.10+. Run everything from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

```bash
# 1. source data (needs git)
python scripts/01_prepare_data.py

# 2. benchmark (v1 = seed 42; v2-v5 = seeds 7, 123, 2026, 31415)
python scripts/02_sample_benchmark.py
python scripts/02_sample_benchmark.py --seed 7 --out benchmark_v2.csv

# 3. model answers (API keys: copy example.env to .env and fill it in)
python scripts/03_generate.py --backend dummy --name dummy --limit 5          # dry run, then delete data/generations/dummy.csv
python scripts/03_generate.py --backend anthropic --model claude-haiku-5-5 --name claude-haiku
#    Qwen / Llama: notebooks/run_open_models.ipynb on a Colab GPU

# 4. annotation sheet -> label it -> check it
python scripts/04_annotation_sheets.py build
python scripts/04_annotation_sheets.py check

# 5. metrics
python scripts/05_evaluate.py
python scripts/05_evaluate.py --demo      # example output on synthetic data
```

Step-by-step details: [`docs/running_models.md`](docs/running_models.md).

---

## Data sources and licences

| Dataset | Use | Licence |
|---|---|---|
| SQuAD 2.0 — Rajpurkar, Jia & Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD*, ACL 2018 | English passages/questions, unanswerable questions | CC BY-SA 4.0 |
| UQA — Arif et al., *UQA: Corpus for Urdu Question Answering*, LREC-COLING 2024 | Urdu translation of SQuAD 2.0 | derived from SQuAD 2.0 (CC BY-SA 4.0) |
| LEGAL-UQA — *A Low-Resource Urdu-English Dataset for Legal Question Answering*, [arXiv:2410.13013](https://arxiv.org/abs/2410.13013) | parallel legal QA | MIT |

Because the benchmark contains text derived from SQuAD 2.0 / UQA, the files in `data/` are released under **CC BY-SA 4.0** with attribution to the sources above.

## Authors

Muhammad Hamza Naveed, Abdul Moiz Azher — MS Artificial Intelligence, FAST-NUCES (course project, DS5004 Machine Learning for Data Science).
