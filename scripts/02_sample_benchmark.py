"""
UrduHaluBench - Step 1: build the evaluation sample (benchmark_v1)

Creates ~350 parallel English-Urdu RAG items:
  UQA       : 175 answerable + 75 unanswerable  (SQuAD 2.0 translations)
  LEGAL-UQA :  70 correct article + 30 wrong article (noisy context -> should abstain)

Rules
  * one question per passage, spread evenly across topics (round-robin over titles)
  * contexts capped in length so small open models can handle them
  * fixed random seed -> the sample is reproducible

Run after scripts/01_prepare_data.py:
    python scripts/02_sample_benchmark.py                                  # seed 42 -> benchmark_v1.csv
    python scripts/02_sample_benchmark.py --seed 7 --out benchmark_v2.csv  # a different sample

Benchmarks used in the project (wrong-context rows verified by hand):
    v1 = seed 42 | v2 = seed 7 | v3 = seed 123 | v4 = seed 2026 | v5 = seed 31415
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_UQA_ANSWERABLE = 175
N_UQA_UNANSWERABLE = 75
N_LEGAL_CORRECT = 70
N_LEGAL_WRONG = 30
MAX_CTX_UQA = 2000     # characters (English context)
MAX_CTX_LEGAL = 2500

# (question row_id, replacement article context_index) pairs rejected in manual checking
# because the "wrong" article still lets the question be answered. They get re-drawn.
REJECTED_WRONG_CONTEXT = {
    (365, 188),  # "Who is the head of the Supreme Court?" vs Acting Chief Justice (Art. 180)
    (554, 197),  # "Can the Court regulate its own procedures?" vs Review by Supreme Court (Art. 188)
}
# UQA questions removed after the translation check (Step 2): the Urdu translation changes the
# meaning, drops the answer, or mistranslates the gold answer. Items containing the machine-
# translation artifact "<unk>" are removed automatically as well.
EXCLUDED_UQA_IDS = {
    "5ad2678ad7d075001a429230",  # question turned into a yes/no question ("dichromates")
    "5ad24ff3d7d075001a428ca8",  # "Earth's mass" translated as "on a large scale"
    "572632ceec44d21400f3dc2f",  # Urdu passage scrambled (Paul Baran / packet switching)
    "5727490bdd62a815002e9a82",  # "expulsion" translated as "recruitment" (answer wrong)
    "5709686c200fba1400367f79",  # Urdu passage drops the answer "Thomson"
    "5727c0402ca10214002d9565",  # "intuition" translated as "awareness" (answer wrong)
    "57302700a23a5019007fce89",  # "Islamist" translated as "Islamic" (answer meaning changed)
}

# seeds whose wrong-context rows have been read and verified by hand
VERIFIED_SEEDS = {42, 7, 123, 2026, 31415}

ROOT = Path(__file__).resolve().parents[1]  # repository root (scripts/ is one level down)
DATA = ROOT / "data"
OUT = DATA / "benchmark" / "benchmark_v1.csv"

COLUMNS = ["item_id", "source", "condition", "expected_behavior",
           "question_en", "question_ur", "context_en", "context_ur",
           "gold_answer_en", "gold_answer_ur", "topic", "source_id", "needs_manual_check"]


def round_robin(df: pd.DataFrame, n: int, group_col: str, rng: np.random.Generator,
                skip_col: str | None = None) -> pd.DataFrame:
    """Pick n rows, cycling through groups so every topic is represented evenly.
    Rows flagged in skip_col are dropped *after* shuffling, so excluding an item only
    shifts the picks inside its own topic and leaves the rest of the sample unchanged."""
    groups = [g.sample(frac=1, random_state=int(rng.integers(1e9)))
              for _, g in df.groupby(group_col)]
    if skip_col:
        groups = [g[~g[skip_col]] for g in groups]
    rng.shuffle(groups)
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g.iloc[i])
        i += 1
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} rows available, need {n}")
    return pd.DataFrame(picked)


def sample_uqa(rng: np.random.Generator) -> pd.DataFrame:
    p = pd.read_csv(DATA / "paired" / "uqa_squad_paired_dev.csv", encoding="utf-8-sig")
    p = p[p["context_en"].str.len() <= MAX_CTX_UQA]

    # one question per passage, and answerable / unanswerable never share a passage
    p = p.sample(frac=1, random_state=SEED).drop_duplicates("context_en")
    p["skip"] = (p["id"].isin(EXCLUDED_UQA_IDS)
                 | p[["question_ur", "context_ur", "answers_ur"]].fillna("")
                    .apply(lambda c: c.str.contains("<unk>", regex=False)).any(axis=1))
    ans = round_robin(p[~p["is_impossible"]], N_UQA_ANSWERABLE, "title_en", rng, "skip")
    rest = p[p["is_impossible"] & ~p["context_en"].isin(ans["context_en"])]
    una = round_robin(rest, N_UQA_UNANSWERABLE, "title_en", rng, "skip")

    out = []
    for df, cond in [(ans, "answerable"), (una, "unanswerable")]:
        out.append(pd.DataFrame({
            "source": "uqa",
            "condition": cond,
            "expected_behavior": "answer" if cond == "answerable" else "abstain",
            "question_en": df["question_en"], "question_ur": df["question_ur"],
            "context_en": df["context_en"], "context_ur": df["context_ur"],
            "gold_answer_en": df["answers_en"].fillna("") if cond == "answerable" else "",
            "gold_answer_ur": df["answers_ur"].fillna("") if cond == "answerable" else "",
            "topic": df["title_en"], "source_id": df["id"],
            "needs_manual_check": False,
        }))
    return pd.concat(out, ignore_index=True)


def sample_legal(rng: np.random.Generator) -> pd.DataFrame:
    global SEED
    leg = pd.concat([pd.read_csv(DATA / "legal_uqa" / f, encoding="utf-8-sig")
                     for f in ["legal_uqa_train.csv", "legal_uqa_validation.csv"]], ignore_index=True)
    leg = leg[leg["context_eng"].str.len() <= MAX_CTX_LEGAL].reset_index()
    leg = leg.rename(columns={"index": "row_id"})

    # one question per article
    leg = leg.sample(frac=1, random_state=SEED).drop_duplicates("context_index")
    chosen = leg.iloc[:N_LEGAL_CORRECT + N_LEGAL_WRONG]
    correct, wrong = chosen.iloc[:N_LEGAL_CORRECT], chosen.iloc[N_LEGAL_CORRECT:].copy()

    # wrong-article condition: swap in a different article not used anywhere in the sample
    pool = leg[~leg["context_index"].isin(chosen["context_index"])]
    donors = pool.sample(frac=1, random_state=int(rng.integers(1e9)))
    spare = donors.iloc[len(wrong):]
    donors = donors.iloc[:len(wrong)].copy()
    for k, rid in enumerate(wrong["row_id"]):
        if (rid, donors.iloc[k]["context_index"]) in REJECTED_WRONG_CONTEXT:
            donors.iloc[k] = spare.iloc[0]  # next unused article, deterministic
            spare = spare.iloc[1:]
    wrong["context_eng"] = donors["context_eng"].values
    wrong["context_urdu"] = donors["context_urdu"].values

    out = []
    for df, cond in [(correct, "answerable"), (wrong, "wrong_context")]:
        out.append(pd.DataFrame({
            "source": "legal_uqa",
            "condition": cond,
            "expected_behavior": "answer" if cond == "answerable" else "abstain",
            "question_en": df["question_eng"], "question_ur": df["question_urdu"],
            "context_en": df["context_eng"], "context_ur": df["context_urdu"],
            "gold_answer_en": df["answer_eng"] if cond == "answerable" else "",
            "gold_answer_ur": df["answer_urdu"] if cond == "answerable" else "",
            "topic": "constitution_of_pakistan",
            "source_id": "legal_" + df["row_id"].astype(str),
            # a random article could still answer the question by chance -> verify by hand
            "needs_manual_check": cond == "wrong_context" and SEED not in VERIFIED_SEEDS,
        }))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT.name, help="file name inside data/benchmark/")
    args = ap.parse_args()
    SEED = args.seed
    OUT = OUT.parent / args.out

    rng = np.random.default_rng(SEED)
    bench = pd.concat([sample_uqa(rng), sample_legal(rng)], ignore_index=True)
    bench = bench.sample(frac=1, random_state=SEED).reset_index(drop=True)  # mix sources
    bench["item_id"] = [f"UHB-{i:04d}" for i in range(1, len(bench) + 1)]
    bench = bench[COLUMNS]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"Saved {len(bench)} items -> {OUT.relative_to(ROOT)}\n")
    print(pd.crosstab(bench["source"], bench["condition"], margins=True))
    print(f"\nExpected to abstain: {(bench['expected_behavior'] == 'abstain').mean():.0%}")
    print(f"UQA topics covered:  {bench.loc[bench.source == 'uqa', 'topic'].nunique()} / 35")
    print(f"Rows to check by hand (wrong article): {bench['needs_manual_check'].sum()}")
