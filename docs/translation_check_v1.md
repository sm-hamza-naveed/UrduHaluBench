# Translation check — benchmark_v1 (UQA part)

Date: 1 Nov 2026. Scope: the 250 UQA items of `benchmark_v1.csv` (Urdu = machine translation of SQuAD 2.0 by the UQA authors). LEGAL-UQA items were not checked here (human-verified translations in the original dataset).

## Method
1. **Automatic checks on all 250 items**
   - gold Urdu answer present in the Urdu passage (span check)
   - numbers in the English answer present in the Urdu answer
   - Urdu/English length ratio of passage and question (flag < 0.8 or > 1.5 for passages)
   - share of untranslated Latin-script characters
   - machine-translation artifact token `<unk>`
2. **Manual reading** (bilingual comparison of English and Urdu)
   - all 75 unanswerable questions (checked that negations / altered entities that make them unanswerable survive translation)
   - all 175 answerable gold-answer pairs
   - 45 randomly chosen answerable questions
   - every automatically flagged passage, read side by side

## Results
| Category | Items | Share | Action |
|---|---|---|---|
| No problem / minor style only | 226 | 90.4% | kept |
| Minor error, meaning still recoverable | 14 | 5.6% | kept, listed below |
| Serious error (meaning changed, answer missing or wrong) | 7 | 2.8% | removed and replaced |
| `<unk>` artifact (special character lost) | 3 | 1.2% | removed and replaced |

No unanswerable question became answerable through translation: every negation ("not", "never", "weren't") and every deliberately altered entity was preserved.

### Removed (serious errors) — replaced by the next item from the same topic
| Old ID | SQuAD id | Problem |
|---|---|---|
| UHB-0032 | 5ad2678ad7d075001a429230 | "What can donate dichromates…" turned into a yes/no question |
| UHB-0283 | 5ad24ff3d7d075001a428ca8 | "Earth's mass" translated as "on a large scale" — question meaningless |
| UHB-0189 | 572632ceec44d21400f3dc2f | Urdu passage scrambled (sentences merged out of order) |
| UHB-0307 | 5727490bdd62a815002e9a82 | "expulsion" translated as بھرتی (recruitment) — gold answer wrong |
| UHB-0254 | 5709686c200fba1400367f79 | Urdu passage omits "from manufacturer Thomson" — answer missing in Urdu |
| UHB-0273 | 5727c0402ca10214002d9565 | "intuition" translated as بیداری (awareness) |
| UHB-0203 | 57302700a23a5019007fce89 | "Islamist" translated as اسلامی (Islamic) — meaning changed |
| UHB-0029, UHB-0049, UHB-0193 | (auto) | `<unk>` replaced "≠", "–" and Greek letters |

The 10 replacements were read and are correct. Because of re-ordering, 17 UQA items received new item IDs; LEGAL-UQA items and their IDs are unchanged.

### Kept with minor errors (worth knowing during annotation)
| ID | Issue |
|---|---|
| UHB-0014 | "During which years" → "in which year" |
| UHB-0021 | "Premier" → وزیر اعظم (Prime Minister) |
| UHB-0065 | question word missing, reads like a statement |
| UHB-0073 | ungrammatical question (verb missing) |
| UHB-0090 | starts with جب ("when", relative) instead of کب |
| UHB-0144 | starts with جو instead of کس |
| UHB-0164 | "adaptive" → اپنی مرضی کے مطابق (one answer variant) |
| UHB-0175 | "critically tapered" → تنقیدی طور پر (critical = criticism) |
| UHB-0176 | "neutralize" → غیر جانبدار (politically neutral) |
| UHB-0207 | "interior designers" → داخلہ ڈیزائنرز (admission) |
| UHB-0220 | "Pauli repulsion" → پولی رد عمل (reaction) |
| UHB-0232 | "First Minister" → وزیر اعظم |
| UHB-0248 | English typo "tuititon" transliterated |
| UHB-0319 | "What are antibodies never generated to fight?" → "Which antibodies…" |

Typical style issues (kept): technical terms left in English (e.g. UHF, phylum, NP-complete), literal word order, "1,100" vs "1100".

## For the paper
"A bilingual check of the 250 machine-translated UQA items found serious translation errors in 2.8% and `<unk>` artifacts in 1.2% of items; these were replaced. A further 5.6% contained minor, meaning-preserving errors and were retained."

## Other versions
v2–v5 use the same exclusion list and automatic `<unk>` filter, but were not read manually.
