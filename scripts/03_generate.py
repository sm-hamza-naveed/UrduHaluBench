"""
UrduHaluBench - Step 4: generate model answers for the benchmark (English + Urdu).

Backends
  openai    : any OpenAI-compatible API (OpenAI, Ollama locally, OpenRouter, Groq, ...)
  anthropic : Claude models via the Anthropic API
  hf        : Hugging Face transformers on a GPU (Colab / Kaggle), 4-bit quantised
  dummy     : fake answers, for a dry run of the whole pipeline without any model

Examples
  # dry run (no model, no internet) - checks everything works
  python scripts/03_generate.py --backend dummy --name dummy --limit 10

  # API keys: copy example.env to .env in the repository root and paste your keys there

  # ChatGPT (needs OPENAI_API_KEY in .env)
  python scripts/03_generate.py --backend openai --model gpt-4o-mini --name gpt-4o-mini

  # Claude (needs ANTHROPIC_API_KEY in .env)
  python scripts/03_generate.py --backend anthropic --model claude-haiku-5-5 --name claude-haiku

  # open models on Colab/Kaggle GPU (see notebooks/run_open_models.ipynb)
  python scripts/03_generate.py --backend hf --model Qwen/Qwen3-8B --name qwen3-8b
  python scripts/03_generate.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct --name llama3.1-8b

  # pilot only (the 30 pilot items)
  python scripts/03_generate.py ... --pilot

Run from the repository root. Output: data/generations/<name>.csv  (one row per item x language). Re-running resumes:
rows that already have an answer are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import csv
import os
import re
import time
from pathlib import Path

import pandas as pd

from prompts import build_messages

ROOT = Path(__file__).resolve().parents[1]  # repository root (scripts/ is one level down)
OUT_DIR = ROOT / "data" / "generations"

# API keys: read from .env in the repository root (see example.env). Keys already set in the terminal take priority.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    print("note: python-dotenv not installed - .env file not loaded (pip install -r requirements.txt)")
FIELDS = ["item_id", "language", "model_name", "model_id", "answer", "seconds", "error"]

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0   # greedy / deterministic -> reproducible


# ------------------------------------------------------------------ backends
class DummyBackend:
    def __init__(self, *_):
        self.i = 0

    def __call__(self, messages):
        self.i += 1
        lang_ur = "اردو" in messages[0]["content"]
        if self.i % 4 == 0:
            return "متن میں اس سوال کا جواب موجود نہیں ہے۔" if lang_ur else "The passage does not contain the answer."
        return "یہ ایک فرضی جواب ہے۔" if lang_ur else "This is a dummy answer."


class OpenAIBackend:
    def __init__(self, model, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model
        # reasoning models (gpt-5*, o-series) do not accept temperature / max_tokens
        self.reasoning = bool(re.match(r"^(gpt-5|o\d)", model))

    def __call__(self, messages):
        kw = dict(model=self.model, messages=messages)
        if self.reasoning:
            kw["max_completion_tokens"] = 2000
        else:
            kw.update(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS, seed=0)
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limits / network: back off and retry
                if attempt == 5:
                    raise
                wait = 2 ** attempt * 5
                print(f"  API error ({type(e).__name__}): retrying in {wait}s")
                time.sleep(wait)


class EmptyAnswer(Exception):
    """The model returned no text; recorded in the 'error' column so the row is redone on re-run."""


class AnthropicBackend:
    """Claude models via the Anthropic API (needs ANTHROPIC_API_KEY in .env)."""

    def __init__(self, model, *_):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.use_temperature = True

    def __call__(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        max_tokens = MAX_NEW_TOKENS
        for attempt in range(6):
            kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=chat)
            if self.use_temperature:
                kw["temperature"] = TEMPERATURE
            try:
                r = self.client.messages.create(**kw)
                text = "".join(getattr(b, "text", "") for b in r.content if b.type == "text").strip()
                if text:
                    return text
                # no visible text: usually the token limit was used up before the answer (e.g. by
                # internal thinking). Retry once with a larger limit, otherwise report why.
                if r.stop_reason == "max_tokens" and max_tokens < 2048:
                    max_tokens = 2048
                    continue
                raise EmptyAnswer(f"empty answer: stop_reason={r.stop_reason}, "
                                  f"blocks={[b.type for b in r.content]}")
            except EmptyAnswer:
                raise
            except Exception as e:
                # some newer models reject a fixed temperature: drop it once and retry immediately
                if self.use_temperature and "temperature" in str(e).lower():
                    print("  model does not accept temperature - using its default sampling")
                    self.use_temperature = False
                    continue
                if attempt == 5:
                    raise
                wait = 2 ** attempt * 5
                print(f"  API error ({type(e).__name__}): retrying in {wait}s")
                time.sleep(wait)


class HFBackend:
    def __init__(self, model, *_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant, device_map="auto")
        self.model.eval()

    def __call__(self, messages):
        # enable_thinking=False switches off Qwen3's <think> mode; other templates ignore it
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


BACKENDS = {"dummy": DummyBackend, "openai": OpenAIBackend, "anthropic": AnthropicBackend, "hf": HFBackend}


# --------------------------------------------------------------- pilot items
def pilot_items(bench: pd.DataFrame, n: int = 30, seed: int = 0) -> list[str]:
    """30 items with the benchmark's mix: ~70% answer / ~30% abstain, both sources."""
    picks = []
    for (src, exp), g in bench.groupby(["source", "expected_behavior"]):
        k = round(n * len(g) / len(bench))
        picks += list(g.sample(min(k, len(g)), random_state=seed)["item_id"])
    return sorted(picks[:n])


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--model", default="dummy", help="model id, e.g. gpt-4o-mini or Qwen/Qwen3-8B")
    ap.add_argument("--name", required=True, help="short name used for the output file")
    ap.add_argument("--base-url", default=None, help="for OpenAI-compatible servers, e.g. http://localhost:11434/v1")
    ap.add_argument("--benchmark", default=str(ROOT / "data" / "benchmark" / "benchmark_v1.csv"))
    ap.add_argument("--languages", nargs="+", default=["en", "ur"])
    ap.add_argument("--pilot", action="store_true", help="only the 30 pilot items")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (testing)")
    args = ap.parse_args()

    bench = pd.read_csv(args.benchmark, encoding="utf-8-sig", keep_default_na=False)
    if args.pilot:
        bench = bench[bench.item_id.isin(pilot_items(bench))]
    if args.limit:
        bench = bench.head(args.limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.name}.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False)
        ok = (prev.error == "") & (prev.answer.astype(str).str.strip() != "")  # empty answers are redone
        done = set(zip(prev.item_id[ok], prev.language[ok]))
    todo = [(r, lang) for _, r in bench.iterrows() for lang in args.languages
            if (r.item_id, lang) not in done]
    print(f"{args.name}: {len(todo)} answers to generate ({len(done)} already done) -> {out.relative_to(ROOT)}")
    if not todo:
        return

    print("loading model ...")
    gen = BACKENDS[args.backend](args.model, args.base_url)

    new_file = not out.exists()
    with open(out, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        n_err = 0
        for k, (r, lang) in enumerate(todo, 1):
            msgs = build_messages(lang, r[f"context_{lang}"], r[f"question_{lang}"])
            t0, ans, err = time.time(), "", ""
            try:
                ans = gen(msgs).strip()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
            if not ans and not err:
                err = "EmptyAnswer: model returned no text"
            n_err += bool(err)
            w.writerow({"item_id": r.item_id, "language": lang, "model_name": args.name,
                        "model_id": args.model, "answer": ans, "seconds": round(time.time() - t0, 2),
                        "error": err})
            f.flush()  # every answer is saved immediately
            if k % 10 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  last: [{lang}] {r.item_id} -> {ans[:60]!r}{'  ERROR' if err else ''}")
    if n_err:
        print(f"{n_err} answers failed or came back empty (reason in the 'error' column). "
              f"Run the same command again to redo only those.")
    else:
        print("all answers generated successfully")


if __name__ == "__main__":
    main()
