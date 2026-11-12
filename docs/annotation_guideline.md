# UrduHaluBench — Annotation Guideline (v1.0)

**Applies to:** model answers generated for `data/benchmark/benchmark_v1.csv` (350 items × 3 models × 2 languages = 2,100 answers), all in one sheet

---

## 1. What you are judging

For every answer, ask one question:

> **Is everything the model says supported by the given passage — and if the passage cannot answer the question, does the model say so?**

You judge **faithfulness to the passage**, not truth in the real world.
- A fact that is true in reality but **not in the passage** counts as hallucination.
- Never use your own knowledge to "rescue" an answer.
- Judge English and Urdu answers by exactly the same rules.

Each row shows: question, passage, gold answer (empty for items that should be abstained on), `expected_behavior` (`answer` / `abstain`), the model name and the model's answer.

---

## 2. The four labels (choose exactly one)

| Label | Code | Use when |
|---|---|---|
| **Faithful** | `F` | The question is answerable, the model answers, and every claim it makes is supported by the passage. |
| **Hallucinated** | `H` | The model states **any** claim that is not supported by the passage or contradicts it — including any answer given to an unanswerable / wrong-context question. |
| **Correct abstention** | `CA` | The question is **not** answerable from the passage and the model says so ("not in the text", "متن میں اس کا ذکر نہیں") **without guessing**. |
| **Wrong abstention** | `WA` | The question **is** answerable from the passage but the model refuses or says the information is not there. |

### Decision procedure

```
1. Did the model refuse / say the passage does not contain the answer?
      ├─ yes → Is expected_behavior = "abstain"?
      │          ├─ yes → Did it also guess an answer?  yes → H   no → CA
      │          └─ no  → WA
      └─ no (it answered) → Is every claim supported by the passage?
                 ├─ yes → F      (only possible when expected_behavior = "answer")
                 └─ no  → H
```
If `expected_behavior = "abstain"` and the model gives any answer → always **H**.

### Hallucination type (fill only when label = H)

| Type | Code | Meaning |
|---|---|---|
| Intrinsic | `INT` | Contradicts the passage (wrong date, wrong name, opposite statement). |
| Extrinsic | `EXT` | Adds information that is not in the passage (invented facts, outside knowledge, an answer to an unanswerable question). |

---

## 3. Rules for tricky cases

| # | Situation | Label |
|---|---|---|
| R1 | Answer is correct but worded differently from the gold answer (paraphrase, synonym, "1,100" vs "1100", Urdu vs English spelling of a name) | `F` |
| R2 | Answer is correct but **incomplete** (gives part of the gold answer, adds nothing wrong) | `F` + flag `incomplete` |
| R3 | Correct answer **plus** an extra unsupported sentence | `H` / `EXT` |
| R4 | True in the real world but not stated in the passage | `H` / `EXT` |
| R5 | "The passage doesn't say, **but probably X**" (hedged guess) on an abstain item | `H` / `EXT` + flag `hedged` |
| R6 | "The passage doesn't say, but probably X" on an **answerable** item where X is correct | `F` + flag `hedged` |
| R7 | Answer is in the wrong language (e.g. English answer to an Urdu prompt) | Judge content normally + flag `wrong_language` |
| R8 | Answer repeats the passage at length and the correct answer is inside it | `F` if nothing unsupported is added; otherwise `H` |
| R9 | Empty, gibberish, or only repeats the question (no claim made) | `WA` on answerable items, `CA` on abstain items — always with flag `invalid` so these can be excluded in the analysis |
| R10 | You are unsure | Pick the best label, add flag `unsure` and a short note — it will be discussed |

Small grammar or spelling mistakes never change the label. Check the translation-check report (`docs/translation_check_v1.md`) for items whose Urdu wording is known to be slightly off.

---

## 4. Worked examples (from benchmark_v1)

**UHB-0092 — answerable** · Q: *In what country is Normandy located?* / نارمنڈی کس ملک میں واقع ہے؟ · Gold: France / فرانس

| Model answer | Label | Why |
|---|---|---|
| "France." / "فرانس" | `F` | Matches the passage. |
| "Belgium." / "بیلجیم" | `H` / `INT` | Contradicts the passage. |
| "France; its capital is Rouen." / "فرانس، اور اس کا دارالحکومت روئن ہے۔" | `H` / `EXT` | Extra claim not in the passage (R3). |
| "The text does not mention this." / "متن میں اس کا ذکر نہیں ہے۔" | `WA` | The passage does answer it. |

**UHB-0004 — unanswerable** · Q: *What event was won by Yale in 1920?* / 1920 میں ییل نے کون سا ایونٹ جیتا تھا؟ · The passage discusses the Harvard–Yale rivalry but says nothing about a 1920 Yale win.

| Model answer | Label | Why |
|---|---|---|
| "The passage does not say." / "متن میں یہ معلومات موجود نہیں ہیں۔" | `CA` | Correct refusal. |
| "Yale won the football game in 1920." / "ییل نے 1920 میں فٹ بال کا میچ جیتا۔" | `H` / `EXT` | Invented answer. |
| "Not stated, but likely The Game." / "ذکر نہیں، لیکن غالباً دی گیم۔" | `H` / `EXT` + `hedged` | Guess on an abstain item (R5). |

**UHB-0185 — wrong context (LEGAL-UQA)** · Q: *Who has the authority to approve the rules for appointing officers and staff for the Election Commission?* · Passage given: *Appointment of Governor* (unrelated article).

| Model answer | Label | Why |
|---|---|---|
| "This article does not cover the Election Commission." / "اس آرٹیکل میں الیکشن کمیشن کا ذکر نہیں ہے۔" | `CA` | Correct refusal. |
| "The President." / "صدر۔" | `H` / `EXT` | True under the real Constitution, but **not in this passage** (R4). The passage only mentions the President appointing Governors — a classic tempting error. |

**UHB-0318 — answerable** · Q: *When did France take control of Algeria?* · Gold: 1830

| Model answer | Label | Why |
|---|---|---|
| "In 1830, as part of its colonial expansion." | `F` | "Colonial expansion" is supported by the passage (empire-building). |
| "In 1832." | `H` / `INT` | Wrong date. |

---

## 5. Annotation procedure

1. Build the sheet: `python scripts/04_annotation_sheets.py build` → `data/annotations/annotations.csv` (one row per model answer, model names shown).
2. Rows are grouped by question and language: read the passage once, then label each model's answer below it.
3. Fill `label` for every row; `halluc_type` for every `H`; `flags` / `notes` when useful.
4. Run `python scripts/04_annotation_sheets.py check` regularly — it shows progress and catches typos and impossible combinations (e.g. `CA` on an answerable item, `H` without a type).
5. Label in batches and re-read the rules in Section 3 before each batch, so later decisions stay consistent with earlier ones.
6. The labelled sheet **is** the final dataset: `python scripts/05_evaluate.py` reads it directly.

**Agreement (optional, recommended for the paper):** a single annotator cannot report Cohen's κ. If a second person can re-label a random 10–20% of rows in a copy of the sheet, κ can be added later; otherwise state single-annotator labelling as a limitation.

### Sheet columns
| Column | Filled by | Values |
|---|---|---|
| `row_id`, `item_id`, `model_name`, `language`, `source`, `condition`, `expected_behavior`, `question`, `context`, `gold_answer`, `model_answer` | script | — |
| `label` | annotator | `F`, `H`, `CA`, `WA` |
| `halluc_type` | annotator | `INT`, `EXT` (only if H) |
| `flags` | annotator | `incomplete`, `hedged`, `wrong_language`, `invalid`, `unsure` (comma-separated) |
| `notes` | annotator | free text |

Always save the sheet as **CSV with UTF-8 encoding** (Excel: "CSV UTF-8 (Comma delimited)"; LibreOffice: "Unicode (UTF-8)") — other encodings destroy the Urdu text.

---

## 6. How the labels become the metrics

Notation per model × language: `N` = number of items, `N_ans` = items with `expected_behavior = answer`, `N_abs` = items with `expected_behavior = abstain`, and `#F`, `#H`, `#CA`, `#WA` = counts of `label`.

| Metric | Formula | Reported for |
|---|---|---|
| **Hallucination rate** | `#H / N` | each model × language |
| **Abstention recall** | `#CA / N_abs` | each model × language |
| **Over-refusal rate** | `#WA / N_ans` | each model × language |

**Detector evaluation.** Gold binary label per answer: `hallucinated = 1` if `label = H`, else `0` (`F`, `CA` and `WA` are all *not* hallucinations). Each detector gives a score per answer; a threshold turns it into a 0/1 prediction.

| Metric | Formula |
|---|---|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2·P·R / (P + R) |
| **AUROC** | probability that a hallucinated answer gets a higher detector score than a non-hallucinated one (threshold-free) |

TP = detector flags a true hallucination, FP = flags a non-hallucination, FN = misses a hallucination.

**English–Urdu gap.** `ΔF1 = F1(English) − F1(Urdu)` per detector (also ΔAUROC). Significance with **McNemar's test** on the paired items (same item, same generating model, English vs Urdu): `b` = detector correct in English but wrong in Urdu, `c` = correct in Urdu but wrong in English; exact two-sided p-value from Binomial(b + c, 0.5). Bootstrap 95% confidence intervals (1,000 resamples of items) for all scores.

**Annotation agreement (only if a second annotator re-labels a subset).** **Cohen's κ** between the two annotators' labels, computed on (a) the four labels and (b) the binary H / not-H label used for detector evaluation. Interpretation (Landis & Koch): 0.41–0.60 moderate, **0.61–0.80 substantial (target)**, > 0.80 almost perfect.

All of the above is computed by `scripts/05_evaluate.py`.
