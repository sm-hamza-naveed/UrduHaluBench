# Running the models

Four models answer every benchmark question in English and in Urdu:

| Model | Type | Where it runs | Cost |
|---|---|---|---|
| Qwen3-8B (`Qwen/Qwen3-8B`) | open | Google Colab / Kaggle (free GPU) | free |
| Llama-3.1-8B-Instruct (`meta-llama/Llama-3.1-8B-Instruct`) | open | Google Colab / Kaggle (free GPU) | free (needs HF access approval) |
| GPT-4o-mini (`gpt-4o-mini`) | API (OpenAI) | locally | well under $1 for the whole benchmark |
| Claude Haiku 5.5 (`claude-haiku-5-5`) | API (Anthropic) | locally | a few dollars at most for the whole benchmark |

All of them get **exactly the same prompt** (`prompts.py`): answer only from the passage, otherwise reply with a fixed "not in the passage" sentence. Decoding is greedy (temperature 0) so runs are reproducible.

## API keys (one-time setup)
1. In the repository root, copy `example.env` and rename the copy to `.env`.
2. Open `.env` in a text editor and paste your keys after the `=` sign (no quotes, no spaces).
3. `pip install -r requirements.txt` (installs `python-dotenv`, which reads `.env`).

`generate.py` loads `.env` automatically. A key already set as an environment variable in the terminal takes priority. `.env` is in `.gitignore` — never share or upload it.

## 0. Dry run (2 minutes, no model needed)
Always run commands from the **repository root**, with the virtual environment active:
```
python scripts/03_generate.py --backend dummy --name dummy --limit 5
```
If that writes `data/generations/dummy.csv`, everything is set up. **Delete `dummy.csv` afterwards** so it is not treated as a model.

## 1. Pilot first (30 items)
Run every model with `--pilot` before the full benchmark. That gives 60 answers per model to check before the full run.

### GPT-4o-mini (locally)
1. Create an API key at https://platform.openai.com/api-keys (add a small credit, e.g. $5).
2. Put the key in `.env` as `OPENAI_API_KEY=...`, then run:
   ```
   python scripts/03_generate.py --backend openai --model gpt-4o-mini --name gpt-4o-mini --pilot
   ```
3. If `gpt-4o-mini` has been retired, use `--model gpt-5-mini --name gpt-5-mini` instead (a reasoning model; the script handles its different settings).

### Claude (locally)
1. Create an API key at https://console.anthropic.com/settings/keys (add a small credit, e.g. $5).
2. Put the key in `.env` as `ANTHROPIC_API_KEY=...`, then run:
   ```
   python scripts/03_generate.py --backend anthropic --model claude-haiku-5-5 --name claude-haiku --pilot
   ```
3. Haiku is the cheapest Claude model. For a stronger one use `--model claude-sonnet-5-5 --name claude-sonnet` (more expensive). If a model refuses the fixed temperature setting, the script notices and switches to its default automatically — mention that in the paper.

### Qwen3-8B and Llama-3.1-8B (on Colab)
Open `notebooks/run_open_models.ipynb` in Google Colab and follow the cells (T4 GPU, Drive folder, Hugging Face token). About 5 minutes per model for the pilot.

### Check the pilot answers
Open the four files in `data/generations/` and read ~10 answers per model:
- short answers, in the right language?
- Qwen must not show `<think>` text?
- the `error` column empty?

Then build the labelling sheet (see Step 5 below).

## 2. Full benchmark (after checking the pilot answers)
Same commands **without** `--pilot`. Already-generated pilot answers are kept and skipped.

| Model | Answers | Approx. time |
|---|---|---|
| GPT-4o-mini | 700 | 15–30 min |
| Qwen3-8B on T4 | 700 | ~1 h |
| Llama-3.1-8B on T4 | 700 | ~1 h |
| Claude Haiku | 700 | 15–30 min |

## 3. Labelling sheet
```
python scripts/04_annotation_sheets.py build
```
→ **one** sheet, `data/annotations/annotations.csv`, with every answer of every model (real model names shown). Rows are grouped by question: read the passage once, label the model answers below it. Fill `label`, `halluc_type`, `flags`, `notes` (rules in `docs/annotation_guideline.md`). The labelled sheet is the final dataset.

- Progress and mistakes: `python scripts/04_annotation_sheets.py check`
- Results at any time (unlabelled rows are skipped): `python scripts/05_evaluate.py`
- Running `build` again (e.g. after adding a model) keeps all labels already entered.
- Save the sheet as **CSV with UTF-8 encoding** (Excel: "CSV UTF-8 (Comma delimited)"); other encodings destroy the Urdu. `check` warns if that happened.

## Good to know
- **Interrupted?** Just run the same command again — it continues from where it stopped.
- **Errors** (rate limits, network) are retried automatically; failed rows are written with the error message and redone on the next run.
- **Optional, local open models:** if your machine has an NVIDIA GPU (8 GB+), you can run the open models with [Ollama](https://ollama.com) instead of Colab:
  `ollama pull qwen3:8b` then
  `python scripts/03_generate.py --backend openai --base-url http://localhost:11434/v1 --model qwen3:8b --name qwen3-8b`
  (Ollama needs no real key, but the OpenAI library wants one: put `OPENAI_API_KEY=ollama` in `.env`). Note the quantisation then differs slightly from the Colab runs — use one route per model, not both.

## Files
All scripts are in `scripts/`, numbered in the order they are run.

| File | Purpose |
|---|---|
| `prompts.py` | the fixed English and Urdu prompts |
| `generate.py` | runs a model over the benchmark → `data/generations/<name>.csv` |
| `run_open_models.ipynb` | Colab notebook for Qwen and Llama |
| `annotation_sheets.py` | `build` the single labelling sheet / `check` progress and mistakes |
| `scripts/05_evaluate.py` | computes all metrics from the labelled sheet |
