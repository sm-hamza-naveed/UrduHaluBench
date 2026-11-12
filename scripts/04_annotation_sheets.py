"""
UrduHaluBench - one annotation sheet for all model answers.

  python scripts/04_annotation_sheets.py build    # create / update the sheet
  python scripts/04_annotation_sheets.py check    # progress + mistakes in your labels

The sheet is  data/annotations/annotations.csv  - you label it, and the labelled file IS the final
dataset that scripts/05_evaluate.py reads:
  python scripts/05_evaluate.py --file data/annotations/annotations.csv

Rows are sorted by question, then language, then model: you read a passage once and label the
three model answers below it. Model names are shown.

Columns you fill in:  label (F / H / CA / WA), halluc_type (INT / EXT, only for H), flags, notes.

Running `build` again later (e.g. after adding a model) keeps every label you already entered
and only adds the new answers.

Always save the sheet as CSV with UTF-8 encoding (Excel: "CSV UTF-8 (Comma delimited)") - other
encodings destroy the Urdu text.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repository root (scripts/ is one level down)
GEN = ROOT / "data" / "generations"
SHEET = ROOT / "data" / "annotations" / "annotations.csv"
BENCH = ROOT / "data" / "benchmark" / "benchmark_v1.csv"

LABELS = {"F", "H", "CA", "WA"}
TYPES = {"INT", "EXT"}
FLAGS = {"incomplete", "hedged", "wrong_language", "invalid", "unsure"}
INPUT_COLS = ["label", "halluc_type", "flags", "notes"]
COLS = ["row_id", "item_id", "model_name", "language", "source", "condition", "expected_behavior",
        "question", "context", "gold_answer", "model_answer"] + INPUT_COLS
KEY = ["item_id", "language", "model_name"]


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype=str)


def load_generations() -> pd.DataFrame:
    files = sorted(GEN.glob("*.csv"))
    if not files:
        raise SystemExit(f"no generation files in {GEN}")
    g = pd.concat([read_csv(f) for f in files], ignore_index=True)
    g = g[(g.error == "") & (g.answer.str.strip() != "")]                 # skip failed / empty answers
    g = g.drop_duplicates(KEY, keep="last")                                # latest successful answer
    print("answers per model:", g.groupby("model_name").size().to_dict())
    return g


def build():
    bench = read_csv(BENCH)
    m = load_generations().merge(bench, on="item_id", how="left", validate="many_to_one")
    pick = lambda col: m[f"{col}_en"].where(m.language == "en", m[f"{col}_ur"])
    new = pd.DataFrame({
        "item_id": m.item_id, "model_name": m.model_name, "language": m.language,
        "source": m.source, "condition": m.condition, "expected_behavior": m.expected_behavior,
        "question": pick("question"), "context": pick("context"), "gold_answer": pick("gold_answer"),
        "model_answer": m.answer, **{c: "" for c in INPUT_COLS},
    })

    if SHEET.exists():  # keep labels already entered
        old = read_csv(SHEET)
        labelled = old[KEY + INPUT_COLS]
        new = new.drop(columns=INPUT_COLS).merge(labelled, on=KEY, how="left").fillna("")
        changed = old.merge(new[KEY + ["model_answer"]], on=KEY, suffixes=("_old", ""))
        changed = changed[(changed.model_answer_old != changed.model_answer) & (changed.label != "")]
        if len(changed):
            print(f"WARNING: {len(changed)} labelled answers have changed since labelling "
                  f"(regenerated?) - re-check rows: {list(changed.row_id[:10])}")
        n_kept = (new.label != "").sum()
    else:
        n_kept = 0

    lang_order = {"en": 0, "ur": 1}
    new = new.sort_values(["item_id", "language", "model_name"],
                          key=lambda s: s.map(lang_order) if s.name == "language" else s).reset_index(drop=True)
    new.insert(0, "row_id", [f"R{i:05d}" for i in range(1, len(new) + 1)])
    new = new[COLS]

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"sheet: {SHEET.relative_to(ROOT)}  |  {len(new)} answers  |  "
          f"{new.item_id.nunique()} questions x {new.language.nunique()} languages x "
          f"{new.model_name.nunique()} models  |  labels kept: {n_kept}")


def check():
    s = read_csv(SHEET)
    lab = s.label.str.strip().str.upper()
    typ = s.halluc_type.str.strip().str.upper()
    done = lab != ""
    print(f"progress: {done.sum()} / {len(s)} labelled ({done.mean():.0%})")
    if done.any():
        print("by language:", s[done].groupby("language").size().to_dict(),
              "| by model:", s[done].groupby("model_name").size().to_dict())
        print("label counts:", lab[done].value_counts().to_dict())

    problems = {
        "unknown label (use F, H, CA, WA)": done & ~lab.isin(LABELS),
        "H without halluc_type (INT / EXT)": (lab == "H") & (typ == ""),
        "halluc_type given but label is not H": (typ != "") & (lab != "H") & done,
        "unknown halluc_type": (typ != "") & ~typ.isin(TYPES),
        "F on an item that should be abstained on": (lab == "F") & (s.expected_behavior == "abstain"),
        "CA on an answerable item (should be WA or F/H)": (lab == "CA") & (s.expected_behavior == "answer"),
        "WA on an abstain item (should be CA or H)": (lab == "WA") & (s.expected_behavior == "abstain"),
        "unknown flag": s["flags"].str.strip().ne("") & ~s["flags"].apply(
            lambda f: all(x.strip().lower() in FLAGS for x in f.split(",") if x.strip())),
        # Excel "CSV" (not "CSV UTF-8") turns Urdu into ??? - detect it
        "Urdu text damaged (saved as plain CSV?)": (s.language == "ur") & s.question.str.contains(r"\?{3,}"),
    }
    bad = {k: v for k, v in problems.items() if v.any()}
    if not bad:
        print("no problems found")
    for k, v in bad.items():
        print(f"- {k}: {v.sum()} rows, e.g. {list(s.row_id[v][:8])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    a = ap.parse_args()
    build() if a.cmd == "build" else check()
