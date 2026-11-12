# How to Annotate

**Your job:** read each model answer and give it one label.

**Golden rule:** judge the answer **only against the passage**. If the passage doesn't say it, it counts as made up, even if it's true in real life.

---

## The 4 labels

| Label | When | Example |
|---|---|---|
| **F** (Faithful) | The model answered, and the answer is in the passage | Q: Where is Normandy? → "France" ✅ |
| **H** (Hallucinated) | The model said something **not** in the passage, or wrong | → "Belgium" ❌ |
| **CA** (Correct abstention) | `expected_behavior` = **abstain**, and the model said "not in the passage" | → "The passage does not contain the answer." ✅ |
| **WA** (Wrong abstention) | `expected_behavior` = **answer**, but the model said "not in the passage" | → "The passage does not contain the answer." ❌ |

## Pick the label in 2 questions

**1. Did the model say "not in the passage"?**
- Yes, and `expected_behavior` = abstain → **CA**
- Yes, but `expected_behavior` = answer → **WA**

**2. Did it give an answer instead?**
- Everything it says is in the passage → **F**
- Anything not in the passage, or wrong → **H**
- `expected_behavior` = abstain and it answered anyway → always **H**

## For every H, fill `halluc_type`
- **INT** = contradicts the passage (wrong year, wrong name)
- **EXT** = adds something not in the passage (an invented answer, outside knowledge)

## Tricky cases
| Situation | Label |
|---|---|
| Right answer, different words ("1,100" vs "1100") | **F** |
| Right answer + an extra fact not in the passage | **H** (EXT) |
| "Not in the passage, but probably X" | **H** (EXT), flag `hedged` |
| True in real life but not in the passage | **H** (EXT) |
| Answer in the wrong language | Judge normally, flag `wrong_language` |
| Not sure | Best guess, flag `unsure` |

Ignore small grammar or spelling mistakes.

---

## Steps
1. Open `data/annotations/annotations.csv` in a spreadsheet app (LibreOffice Calc, Excel, Google Sheets).
2. For each row read `question`, `context` and `model_answer`, then fill `label` (and `halluc_type` for H).
3. Save it as **CSV with UTF-8 encoding**, otherwise the Urdu text is destroyed (Excel: "CSV UTF-8 (Comma delimited)"; LibreOffice: character set "Unicode (UTF-8)").
4. Check your work: `python scripts/04_annotation_sheets.py check`

Rows are grouped by question: read the passage once, then label the 3 model answers below it.
