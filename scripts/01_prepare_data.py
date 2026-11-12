"""
UrduHaluBench - data preparation
Downloads UQA (Urdu), SQuAD 2.0 dev (English) and LEGAL-UQA, loads them into
pandas DataFrames, builds the Urdu-English paired file and saves CSVs to data/.

Run from the project folder (with the virtual environment activated):
    python scripts/01_prepare_data.py

Requires: git, pandas, pyarrow
"""
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]  # repository root (scripts/ is one level down)
RAW = ROOT / "raw"    # cloned repositories
DATA = ROOT / "data"  # cleaned CSVs

REPOS = {
    "UQA": "https://github.com/sameearif/UQA.git",
    "LEGAL-UQA": "https://github.com/nlp-anonymous-researcher/LEGAL-UQA.git",
}


# ---------------------------------------------------------------- 1. download
def clone_repos():
    RAW.mkdir(exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")  # skip the 178 MB UQA train file
    for name, url in REPOS.items():
        target = RAW / name
        if target.exists():
            print(f"[skip] {name} already downloaded")
            continue
        print(f"[clone] {url}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True, env=env)


# ---------------------------------------------------- 2. arrow -> DataFrame
def read_arrow(path: Path) -> pd.DataFrame:
    """Read a HuggingFace `save_to_disk` .arrow file into a pandas DataFrame."""
    with pa.memory_map(str(path), "r") as source:
        try:
            table = pa.ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(source).read_all()
    return table.to_pandas()


def load_uqa() -> pd.DataFrame:
    df = read_arrow(RAW / "UQA" / "UQA" / "validation" / "data-00000-of-00001.arrow")
    print(f"UQA validation: {len(df):,} rows, {df['id'].nunique():,} unique questions")
    return df


def load_squad_dev() -> pd.DataFrame:
    with open(RAW / "UQA" / "SQuAD" / "dev-v2.0.json", encoding="utf-8") as f:
        squad = json.load(f)
    rows = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                rows.append({
                    "id": qa["id"],
                    "title_en": article["title"],
                    "context_en": para["context"],
                    "question_en": qa["question"],
                    "answers_en": " || ".join(answers),
                    "is_impossible_en": qa.get("is_impossible", False),
                })
    df = pd.DataFrame(rows)
    print(f"SQuAD 2.0 dev (English): {len(df):,} questions")
    return df


def load_legal_uqa() -> pd.DataFrame:
    base = RAW / "LEGAL-UQA" / "data" / "complete_dataset"
    parts = []
    for split in ["train", "validation"]:
        part = read_arrow(base / split / "data-00000-of-00001.arrow")
        part["split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True).drop(columns=["__index_level_0__"], errors="ignore")
    print(f"LEGAL-UQA: {len(df):,} English-Urdu pairs")
    return df


# ------------------------------------------------- 3. pair Urdu with English
def build_paired(uqa: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    # UQA repeats a question once per alternative gold answer -> one row per question
    uqa_q = (uqa.groupby("id", sort=False)
                .agg(title_ur=("title", "first"),
                     context_ur=("context", "first"),
                     question_ur=("question", "first"),
                     is_impossible=("is_impossible", "first"),
                     answers_ur=("answer", lambda s: " || ".join(dict.fromkeys(a for a in s if a))))
                .reset_index())
    paired = uqa_q.merge(squad, on="id", how="inner", validate="one_to_one")

    assert len(paired) == len(uqa_q), "some Urdu questions have no English original"
    assert (paired["is_impossible"] == paired["is_impossible_en"]).all(), "label mismatch"
    paired = paired.drop(columns="is_impossible_en")

    paired = paired[["id", "is_impossible", "question_en", "question_ur", "context_en", "context_ur",
                     "answers_en", "answers_ur", "title_en", "title_ur"]]
    print(f"Paired: {len(paired):,} questions "
          f"({paired['is_impossible'].sum():,} unanswerable, {paired['context_en'].nunique():,} passages)")
    return paired


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    clone_repos()

    uqa = load_uqa()
    squad = load_squad_dev()
    legal = load_legal_uqa()
    paired = build_paired(uqa, squad)

    articles = pd.read_csv(RAW / "LEGAL-UQA" / "data" / "document_chunks" /
                           "article_wise_chunks_eng_urdu_processed.csv")

    # utf-8-sig keeps Urdu readable when the CSV is opened in Excel
    out = {
        DATA / "paired" / "uqa_squad_paired_dev.csv": paired,
        DATA / "uqa" / "uqa_validation_ur.csv": uqa,
        DATA / "legal_uqa" / "legal_uqa_all.csv": legal,
        # per-split files, read by 02_sample_benchmark.py (row order = train then validation)
        DATA / "legal_uqa" / "legal_uqa_train.csv": legal[legal.split == "train"].drop(columns="split"),
        DATA / "legal_uqa" / "legal_uqa_validation.csv": legal[legal.split == "validation"].drop(columns="split"),
        DATA / "legal_uqa" / "legal_uqa_constitution_articles.csv": articles,
    }
    for path, df in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[saved] {path.relative_to(ROOT)}")
