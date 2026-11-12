"""
UrduHaluBench - evaluation metrics

Computes, from the finished annotation sheet:
  1. Hallucination rate, abstention recall, over-refusal rate   (per model x language)
  2. Detector precision / recall / F1 / AUROC                     (per detector x language)
  3. English-Urdu gap: dF1, dAUROC + McNemar's exact test         (per detector)
  4. Cohen's kappa between the two annotators                     (4-class and binary)
Bootstrap 95% confidence intervals are given for the main scores.

Input: data/annotations/annotations.csv (the sheet from scripts/04_annotation_sheets.py) with
    item_id, model_name, language (en/ur), expected_behavior (answer/abstain), label (F / H / CA / WA)
Rows without a valid label are skipped, so you can run it on a partly labelled sheet.
(Also accepts the two-annotator format: model_code, label_a1, label_a2, final_label -> adds Cohen's kappa.)
and one score column per detector named  det_<name>  (higher = more likely hallucinated,
range 0-1). Use --threshold to set the cut-off for P/R/F1 (default 0.5).

Usage:
    python scripts/05_evaluate.py                       # uses data/annotations/annotations.csv
    python scripts/05_evaluate.py --demo                # runs on synthetic data to show the output
"""
import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["F", "H", "CA", "WA"]
N_BOOT = 1000
RNG = np.random.default_rng(0)


# ----------------------------------------------------------- basic metrics
def prf(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum()); fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def auroc(y_true, score):
    """Mann-Whitney formulation: P(score of a positive > score of a negative), ties count 0.5."""
    y_true, score = np.asarray(y_true, bool), np.asarray(score, float)
    pos, neg = score[y_true], score[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cohen_kappa(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar_exact(correct_en, correct_ur):
    """Exact two-sided McNemar test on paired correctness (same item, EN vs UR)."""
    correct_en, correct_ur = np.asarray(correct_en, bool), np.asarray(correct_ur, bool)
    b = int((correct_en & ~correct_ur).sum())   # right in English, wrong in Urdu
    c = int((~correct_en & correct_ur).sum())   # right in Urdu, wrong in English
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def boot_ci(df, fn):
    """95% bootstrap CI, resampling items (all rows of an item stay together)."""
    items = df["item_id"].unique()
    groups = {i: g for i, g in df.groupby("item_id")}
    vals = []
    for _ in range(N_BOOT):
        pick = RNG.choice(items, size=len(items), replace=True)
        vals.append(fn(pd.concat([groups[i] for i in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi


def fmt(x, ci=None):
    s = f"{x:.3f}"
    return s + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


# ---------------------------------------------------------- report sections
def generation_metrics(df):
    rows = []
    for (m, lang), g in df.groupby(["model_code", "language"]):
        n_abs = (g.expected_behavior == "abstain").sum(); n_ans = (g.expected_behavior == "answer").sum()
        hr = lambda d: (d.final_label == "H").mean()
        ar = lambda d: ((d.final_label == "CA") & (d.expected_behavior == "abstain")).sum() / max(1, (d.expected_behavior == "abstain").sum())
        orr = lambda d: ((d.final_label == "WA") & (d.expected_behavior == "answer")).sum() / max(1, (d.expected_behavior == "answer").sum())
        rows.append({"model": m, "lang": lang, "N": len(g),
                     "hallucination_rate": fmt(hr(g), boot_ci(g, hr)),
                     "abstention_recall": fmt(ar(g), boot_ci(g, ar)),
                     "over_refusal_rate": fmt(orr(g), boot_ci(g, orr))})
    return pd.DataFrame(rows)


def detector_metrics(df, detectors, thr):
    rows, gap_rows = [], []
    y = df.final_label.eq("H")
    for d in detectors:
        per_lang = {}
        for lang, g in df.groupby("language"):
            yt, sc = g.final_label.eq("H"), g[d]
            p, r, f = prf(yt, sc >= thr)
            a = auroc(yt, sc)
            f_ci = boot_ci(g, lambda x: prf(x.final_label.eq("H"), x[d] >= thr)[2])
            a_ci = boot_ci(g, lambda x: auroc(x.final_label.eq("H"), x[d]))
            per_lang[lang] = (f, a)
            rows.append({"detector": d.removeprefix("det_"), "lang": lang, "precision": f"{p:.3f}",
                         "recall": f"{r:.3f}", "F1": fmt(f, f_ci), "AUROC": fmt(a, a_ci)})
        if {"en", "ur"} <= set(per_lang):
            key = ["item_id", "model_code"]
            en = df[df.language == "en"].set_index(key); ur = df[df.language == "ur"].set_index(key)
            both = en.join(ur, lsuffix="_en", rsuffix="_ur", how="inner")
            ok_en = (both[f"{d}_en"] >= thr) == both["final_label_en"].eq("H")
            ok_ur = (both[f"{d}_ur"] >= thr) == both["final_label_ur"].eq("H")
            b, c, pval = mcnemar_exact(ok_en, ok_ur)
            gap_rows.append({"detector": d.removeprefix("det_"),
                             "dF1 (en-ur)": f"{per_lang['en'][0] - per_lang['ur'][0]:+.3f}",
                             "dAUROC (en-ur)": f"{per_lang['en'][1] - per_lang['ur'][1]:+.3f}",
                             "pairs": len(both), "b (en ok, ur wrong)": b, "c (ur ok, en wrong)": c,
                             "McNemar p": f"{pval:.4f}"})
    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


def agreement(df):
    if not {"label_a1", "label_a2"} <= set(df.columns):
        return None
    d = df.dropna(subset=["label_a1", "label_a2"])
    d = d[(d.label_a1.astype(str).str.strip() != "") & (d.label_a2.astype(str).str.strip() != "")]
    if d.empty:
        return None
    k4 = cohen_kappa(d.label_a1, d.label_a2)
    k2 = cohen_kappa(d.label_a1.eq("H"), d.label_a2.eq("H"))
    return len(d), (d.label_a1 == d.label_a2).mean(), k4, k2


# ----------------------------------------------------------------- demo data
def demo_data(n_items=120, seed=1):
    r = np.random.default_rng(seed); rows = []
    for i in range(n_items):
        exp = "abstain" if r.random() < 0.3 else "answer"
        for m in ["M1", "M2", "M3"]:
            for lang in ["en", "ur"]:
                p_h = (0.15 if exp == "answer" else 0.45) + (0.10 if lang == "ur" else 0)
                if exp == "answer":
                    lab = r.choice(["F", "H", "WA"], p=[1 - p_h - 0.05, p_h, 0.05])
                else:
                    lab = r.choice(["CA", "H"], p=[1 - p_h, p_h])
                a2 = lab if r.random() < 0.88 else r.choice(LABELS)
                h = lab == "H"
                rows.append({"item_id": f"UHB-{i:04d}", "model_code": m, "language": lang,
                             "expected_behavior": exp, "label_a1": lab, "label_a2": a2, "final_label": lab,
                             "det_hhem": float(np.clip(r.normal(0.7 if h else 0.3, 0.2 if lang == "en" else 0.3), 0, 1)),
                             "det_ragas": float(np.clip(r.normal(0.65 if h else 0.35, 0.25), 0, 1)),
                             "det_llm_judge": float(np.clip(r.normal(0.75 if h else 0.25, 0.2), 0, 1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).resolve().parents[1] / "data" / "annotations" / "annotations.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    args = ap.parse_args()

    df = demo_data() if args.demo else pd.read_csv(args.file, encoding="utf-8-sig", keep_default_na=False)
    if args.demo:
        print("*** DEMO MODE: synthetic data, numbers are meaningless ***\n")
    # single-annotator sheet from annotation_sheets.py: 'label' + 'model_name'
    if "final_label" not in df and "label" in df:
        df["final_label"] = df["label"]
    if "model_code" not in df and "model_name" in df:
        df["model_code"] = df["model_name"]
    df["final_label"] = df["final_label"].astype(str).str.strip().str.upper()
    unlabelled = ~df["final_label"].isin(LABELS)
    if unlabelled.any():
        print(f"note: {unlabelled.sum()} of {len(df)} rows have no valid label yet and are left out\n")
        df = df[~unlabelled].copy()
    if df.empty:
        raise SystemExit("no labelled rows to evaluate")
    df["language"] = df["language"].str.lower().str[:2]
    detectors = [c for c in df.columns if c.startswith("det_")]
    pd.set_option("display.width", 200, "display.max_columns", 20)

    print("=== 1. Generation: hallucination & abstention (95% CI) ===")
    print(generation_metrics(df).to_string(index=False), "\n")

    if detectors:
        det, gap = detector_metrics(df, detectors, args.threshold)
        print(f"=== 2. Detectors (threshold {args.threshold}; positive class = hallucinated) ===")
        print(det.to_string(index=False), "\n")
        print("=== 3. English-Urdu gap with McNemar's exact test ===")
        print(gap.to_string(index=False), "\n")

    agr = agreement(df)
    print("=== 4. Annotation agreement (double-labelled rows) ===")
    if agr:
        n, raw, k4, k2 = agr
        print(f"rows: {n} | raw agreement: {raw:.3f} | Cohen's kappa (4 labels): {k4:.3f} | "
              f"Cohen's kappa (H vs not-H): {k2:.3f}")
        print("target: kappa >= 0.61 (substantial)" + ("" if k4 >= 0.61 else "  -> BELOW TARGET: discuss disagreements"))
    else:
        print("single annotator - no agreement score (needs label_a1 and label_a2 columns)")
