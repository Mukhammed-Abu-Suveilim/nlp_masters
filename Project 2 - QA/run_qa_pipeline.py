"""
Legal contract QA pipeline: clean text -> Ollama (qwen3.5:0.8b) -> span alignment -> metrics/submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import ollama
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Config ---
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OLLAMA_MODEL = "qwen3.5:0.8b"
FALLBACK_MODELS = ("qwen2.5:0.5b", "llama3.2:1b")
MAX_CHARS = 4000
HEAD_CHARS = 500
VAL_SIZE = 0.2
RANDOM_STATE = 42
TEMPERATURE = 0.1
NUM_PREDICT = 256

QUESTION_RE = re.compile(
    r'related to ""([^"]+)"".*?Details:\s*(.+)$',
    re.DOTALL,
)
QUICKLINKS_RE = re.compile(r"^QuickLinks\s*--.*$", re.MULTILINE | re.IGNORECASE)
PAGE_LINE_RE = re.compile(r"^\s*(?:-\d+-|\d+)\s*$", re.MULTILINE)

PROMPT_TEMPLATE = """Based ONLY on the contract text below, answer with the shortest possible exact quote from the text.

QUESTION: {question}

CONTRACT TEXT:
{excerpt}

RULES:
- Return ONLY exact words/phrases copied from the contract
- One short phrase or sentence maximum when possible
- If not in the text, reply exactly: NOT FOUND

ANSWER:"""

RUNS_DIR = OUTPUT_DIR / "runs"
CACHE_PATH = OUTPUT_DIR / "cache.jsonl"


def clean_text(text: str) -> str:
    """Clean a copy of context for prompting (original kept for alignment)."""
    if not isinstance(text, str):
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = QUICKLINKS_RE.sub("", out)
    out = PAGE_LINE_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def parse_question(question: str) -> tuple[str, str]:
    m = QUESTION_RE.search(question.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", question.strip()


def _keywords(clause_label: str, detail: str) -> set[str]:
    raw = f"{clause_label} {detail}".lower()
    tokens = re.findall(r"[a-z0-9]+", raw)
    words = {t for t in tokens if len(t) > 2}
    for t in list(words):
        if t.endswith("ies"):
            words.add(t[:-3] + "y")
        if t.endswith("s") and len(t) > 3:
            words.add(t[:-1])
    return words


def retrieve_excerpt(cleaned_context: str, clause_label: str, detail: str) -> str:
    """Select paragraphs most relevant to the clause question."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_context) if p.strip()]
    if not paragraphs:
        return cleaned_context[:MAX_CHARS]

    keywords = _keywords(clause_label, detail)

    def score_para(para: str) -> int:
        low = para.lower()
        return sum(1 for kw in keywords if kw in low)

    scored = sorted(enumerate(paragraphs), key=lambda x: score_para(x[1]), reverse=True)

    head = cleaned_context[:HEAD_CHARS]
    chosen: list[str] = []
    seen: set[int] = set()
    total = len(head)

    if head:
        chosen.append(head)
        seen.add(-1)

    for idx, para in scored:
        if idx in seen:
            continue
        if total + len(para) + 2 > MAX_CHARS:
            continue
        chosen.append(para)
        seen.add(idx)
        total += len(para) + 2
        if total >= MAX_CHARS:
            break

    if len(chosen) <= 1 and paragraphs:
        chosen = [cleaned_context[:MAX_CHARS]]
    return "\n\n".join(chosen)


def build_prompt(question: str, excerpt: str, clause_label: str, detail: str) -> str:
    """Prompt tuned for short extractive answers (see old_code fix for qwen3.5 thinking mode)."""
    del clause_label, detail  # kept in signature for callers; question carries full text
    return PROMPT_TEMPLATE.format(question=question, excerpt=excerpt)


def _slug_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model.replace(":", "-"))


def create_run_dir(model: str, mode: str, limit: int | None = None) -> Path:
    """Create outputs/runs/<timestamp>_<model>_<mode>/ for this execution."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts, _slug_model(model), mode]
    if limit is not None:
        parts.append(f"limit{limit}")
    run_dir = RUNS_DIR / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_config(
    *,
    model: str,
    mode: str,
    use_cache: bool,
    limit: int | None,
    refresh_cache: bool,
    run_dir: Path,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model": model,
        "mode": mode,
        "limit": limit,
        "use_cache": use_cache,
        "refresh_cache": refresh_cache,
        "ollama_options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "top_p": 0.9,
            "think": False,
            "api": "chat (fallback: generate)",
        },
        "retrieval": {
            "max_chars": MAX_CHARS,
            "head_chars": HEAD_CHARS,
        },
        "validation": {
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "paths": {
            "data_dir": str(DATA_DIR),
            "cache_path": str(CACHE_PATH),
        },
    }


def save_run_artifacts(
    run_dir: Path,
    run_config: dict,
    train: pd.DataFrame,
    metrics: dict | None = None,
) -> None:
    """Write prompt template, example prompt, and run config into the run folder."""
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prompt_doc = (
        "# Prompt template\n\n"
        "Placeholders: `{question}` = full question string, `{excerpt}` = retrieved contract text.\n\n"
        "```\n"
        f"{PROMPT_TEMPLATE}\n"
        "```\n"
    )
    (run_dir / "prompt_template.md").write_text(prompt_doc, encoding="utf-8")

    if len(train):
        row = train.iloc[0]
        clause, detail = parse_question(row["question"])
        excerpt = retrieve_excerpt(clean_text(row["context"]), clause, detail)
        example = build_prompt(row["question"], excerpt, clause, detail)
        (run_dir / "prompt_example.txt").write_text(example, encoding="utf-8")

    if metrics is not None:
        merged = {**run_config, **metrics}
        (run_dir / "metrics.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    latest_ptr = OUTPUT_DIR / "latest_run.txt"
    latest_ptr.write_text(str(run_dir.resolve()), encoding="utf-8")


def resolve_ollama_model(preferred: str = OLLAMA_MODEL) -> str:
    """Return preferred model if present locally, else first available fallback."""
    try:
        resp = ollama.list()
        listed = {m.model for m in resp.models}
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama, then run: ollama pull qwen3.5:0.8b"
        ) from exc

    def _has(tag: str) -> bool:
        return tag in listed

    if _has(preferred):
        return preferred
    for fb in FALLBACK_MODELS:
        if _has(fb):
            print(f"Warning: {preferred!r} not found; using fallback {fb!r}")
            return fb
    raise RuntimeError(
        f"Model {preferred!r} not found. Install with: ollama pull {preferred}"
    )


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama; must disable thinking or qwen3.5 returns empty response."""
    options = {
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "top_p": 0.9,
    }
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            think=False,
        )
        text = (response.message.content or "").strip()
    except Exception:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=options,
            think=False,
        )
        text = (response.response or "").strip()

    return text


def normalize_prediction(raw: str) -> str:
    text = raw.strip()
    for prefix in ("ANSWER:", "Answer:", "Exact quote:", "Response:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    text = text.strip("\"'`")
    not_found = (
        "NOT FOUND",
        "NONE",
        "N/A",
        "NA",
        "NOT FOUND IN DOCUMENT",
        "NOT FOUND IN THE DOCUMENT",
    )
    if text.upper() in not_found or any(p in text.lower() for p in (
        "not found in document",
        "cannot find",
        "does not contain",
        "doesn't contain",
    )):
        return ""
    return text


def align_span(prediction: str, original_context: str) -> tuple[str, bool]:
    """Map model output to a verbatim substring of original context."""
    pred = normalize_prediction(prediction)
    if not pred:
        return "", True
    if pred in original_context:
        return pred, True

    alignment = fuzz.partial_ratio_alignment(pred, original_context, score_cutoff=70)
    if alignment is not None:
        start, end = alignment.dest_start, alignment.dest_end
        return original_context[start:end], False

    # Sliding window by length of prediction
    plen = len(pred)
    if plen == 0:
        return "", True
    best_score = 0
    best_span = ""
    step = max(1, plen // 4)
    for i in range(0, max(1, len(original_context) - plen + 1), step):
        window = original_context[i : i + plen]
        score = fuzz.ratio(pred, window)
        if score > best_score:
            best_score = score
            best_span = window
    if best_score >= 60 and best_span:
        return best_span, False
    return pred, False


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


def load_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Skip poisoned entries from runs where think=True left response empty
            if row.get("response"):
                cache[row["key"]] = row["response"]
    return cache


def append_cache(path: Path, key: str, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


def predict_row(
    context: str,
    question: str,
    model: str = OLLAMA_MODEL,
    cache: dict[str, str] | None = None,
    cache_path: Path | None = None,
) -> tuple[str, str, bool]:
    """Returns (aligned_answer, raw_response, was_exact_substring)."""
    clause_label, detail = parse_question(question)
    cleaned = clean_text(context)
    excerpt = retrieve_excerpt(cleaned, clause_label, detail)
    prompt = build_prompt(question, excerpt, clause_label, detail)

    key = _cache_key(model, prompt)
    raw: str | None = None
    if cache is not None and key in cache and cache[key]:
        raw = cache[key]
    if not raw:
        raw = call_ollama(prompt, model=model)
        if cache is not None and raw:
            cache[key] = raw
        if cache_path is not None and raw:
            append_cache(cache_path, key, raw)

    aligned, exact_sub = align_span(raw, context)
    return aligned, raw, exact_sub


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def run_validation(
    train: pd.DataFrame,
    model: str = OLLAMA_MODEL,
    use_cache: bool = True,
    limit: int | None = None,
    cache_path: Path = CACHE_PATH,
) -> tuple[pd.DataFrame, dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path) if use_cache else {}

    train = train.copy()
    train["clause_label"] = train["question"].map(lambda q: parse_question(q)[0])

    idx = range(len(train))
    stratify = None
    if train["clause_label"].nunique() > 1:
        counts = train["clause_label"].value_counts()
        if counts.min() >= 2:
            stratify = train["clause_label"]

    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    val_df = train.iloc[val_idx].reset_index(drop=True)
    if limit is not None:
        val_df = val_df.head(limit).reset_index(drop=True)

    rows = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation"):
        aligned, raw, _ = predict_row(
            row["context"],
            row["question"],
            model=model,
            cache=cache,
            cache_path=cache_path if use_cache else None,
        )
        gold = str(row["answers"])
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "gold": gold,
                "pred_raw": raw,
                "pred": aligned,
                "aligned_verbatim": aligned in row["context"],
                "exact_match": exact_match(aligned, gold),
            }
        )

    pred_df = pd.DataFrame(rows)
    em = pred_df["exact_match"].mean() if len(pred_df) else 0.0
    metrics = {
        "val_exact_match": float(em),
        "n_val": len(pred_df),
        "n_train": len(train_idx),
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return pred_df, metrics


def run_test(
    test: pd.DataFrame,
    model: str = OLLAMA_MODEL,
    use_cache: bool = True,
    limit: int | None = None,
    cache_path: Path = CACHE_PATH,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path) if use_cache else {}

    if limit is not None:
        test = test.head(limit)

    answers = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Test"):
        aligned, _, _ = predict_row(
            row["context"],
            row["question"],
            model=model,
            cache=cache,
            cache_path=cache_path if use_cache else None,
        )
        answers.append(aligned)

    sub = pd.DataFrame({"id": test["id"], "answers": answers})
    return sub


def save_cleaning_samples(train: pd.DataFrame, run_dir: Path, n: int = 3) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "cleaning_samples.md"
    lines = ["# Cleaning samples\n"]
    for i in range(min(n, len(train))):
        raw = train.iloc[i]["context"]
        cleaned = clean_text(raw)
        lines.append(f"## Example {i} (id={train.iloc[i]['id']})\n")
        lines.append("### Before (first 800 chars)\n")
        lines.append(f"```\n{raw[:800]}\n```\n")
        lines.append("### After (first 800 chars)\n")
        lines.append(f"```\n{cleaned[:800]}\n```\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal contract QA pipeline")
    parser.add_argument(
        "--mode",
        choices=["val", "full"],
        default="val",
        help="val: validation only; full: val + test submission",
    )
    parser.add_argument("--model", default=None, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Delete outputs/cache.jsonl before running (fixes empty cached responses)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows for val/test (debug only)",
    )
    args = parser.parse_args()

    model = resolve_ollama_model(args.model or OLLAMA_MODEL)
    print(f"Using Ollama model: {model}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_cache and CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("Deleted stale cache.jsonl")

    run_dir = create_run_dir(model, args.mode, args.limit)
    use_cache = not args.no_cache
    run_config = build_run_config(
        model=model,
        mode=args.mode,
        use_cache=use_cache,
        limit=args.limit,
        refresh_cache=args.refresh_cache,
        run_dir=run_dir,
    )
    print(f"Run directory: {run_dir}")

    train, test = load_data()
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    save_run_artifacts(run_dir, run_config, train)
    save_cleaning_samples(train, run_dir)

    pred_df, metrics = run_validation(
        train, model=model, use_cache=use_cache, limit=args.limit
    )
    if args.limit:
        metrics["note"] = f"limited run (limit={args.limit})"

    pred_df.to_csv(run_dir / "val_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    save_run_artifacts(run_dir, run_config, train, metrics=metrics)
    print(f"Validation exact match: {metrics['val_exact_match']:.4f}")
    print(f"Saved {run_dir / 'metrics.json'} and val_predictions.csv")

    if args.mode == "full":
        sub = run_test(test, model=model, use_cache=use_cache, limit=args.limit)
        sub.to_csv(run_dir / "submission.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {run_dir / 'submission.csv'} ({len(sub)} rows)")

    print(f"All run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
