# Legal Contract QA — Competition & Pipeline Research Brief

## 1. Task overview

This is a **Kaggle-style homework competition** for an NLP course (RUDN). The task is **extractive question answering** on **US legal contracts** (CUAD-style dataset): given a long contract (`context`) and a structured question, predict the **exact text span** from the contract that answers the question.

This is **not** open-ended generation. The leaderboard uses **strict exact match** — predicted and gold strings must match character-for-character (locally we compare with `.strip()` only).

**Homework grading has two parts:**

1. Best **closed (private) leaderboard** exact-match score after the competition ends.
2. **Reproducible code** (preprocessing + training/inference) in a single runnable file matching the submitted solution.

---

## 2. Competition rules (summary)

| Rule | Detail |
|------|--------|
| **Metric** | Exact match between predicted answer and gold answer |
| **Models allowed** | Course models; **local LLMs ≤ 1B parameters**; fine-tuned extractive QA (BERT/SQuAD-style) likely allowed |
| **Models forbidden** | API LLMs (GPT, Gemini, Claude, DeepSeek, etc.) |
| **Extra data** | Allowed for training, **not** the test set |
| **Leaderboards** | Public (during competition) + **private** (final grade); avoid overfitting to public LB |
| **Submission format** | CSV: `id`, `answers` — see `sample_submission.csv` |
| **Baselines** | Several on LB; some are grade thresholds; top-3 get bonus points |

Source: [`Compition overview and rules.txt`](Compition%20overview%20and%20rules.txt)

---

## 3. Dataset

### Files

| File | Rows | Columns |
|------|------|---------|
| `train.csv` | **200** | `id`, `context`, `question`, `answers`, `answer_start` |
| `test.csv` | **165** | `id`, `context`, `question` |
| `sample_submission.csv` | 165 | `id`, `answers` |

### Schema

- **`context`**: Full contract text (often **5k–20k+ characters**), multiline, CSV-quoted. Typical content: exhibits, sponsorship/service agreements, SEC filings, signatures, numbered clauses.
- **`question`**: Fixed template, CUAD-style:

  ```
  Highlight the parts (if any) of this contract related to "<CLAUSE_TYPE>"
  that should be reviewed by a lawyer. Details: <specific sub-question>
  ```

- **`answers`**: **Verbatim substring** of `context` (train only). Can be short (`Bravatek`, `Sponsor`) or long multi-sentence clauses (500+ chars).
- **`answer_start`**: Character offset of the gold span in **original** `context`.

### Domain & difficulty

- **Domain**: US commercial/legal contracts (CUAD-like clause extraction).
- **~41 clause types** (Parties, Effective Date, License Grant, Termination For Convenience, Governing Law, etc.).
- **Noise in text**: doubled quotes `(""ACM"")`, page numbers, `QuickLinks`, signatures `/s/`, inconsistent whitespace, OCR/PDF artifacts.
- **Small train set**: only 200 examples — fine-tuning is possible but high variance; generalization is hard.
- **Answer variability**:
  - Some answers are **entity names** or **dates**.
  - Some are **full sentences/clauses** (exact boundary matters for EM).
  - Some gold spans start at `answer_start=0` (title/header answers like document name from press release text).

### Example (train id=0)

- **Question**: Parties — *"The two or more parties who signed the contract"*
- **Gold answer**: `American Champion Media, Inc.` (offset 107 in context)
- **Context**: Boxing sponsorship agreement between ACM and Shun Li De Commerce & Trading Ltd.

---

## 4. Current pipeline (`run_qa_pipeline.py`)

### Architecture

```
original context
    ├── clean_text() → cleaned copy (for prompting only)
    ├── parse_question() → clause_label + detail
    ├── retrieve_excerpt() → top paragraphs by keyword overlap (~4000 chars max)
    ├── build_prompt() → Ollama chat (qwen3.5:0.8b, think=False)
    ├── normalize_prediction()
    └── align_span() → map output to substring of ORIGINAL context (rapidfuzz)
```

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | `qwen3.5:0.8b` (Ollama, local, ~0.8B params) |
| API | `ollama.chat(..., think=False)` — **required** for Qwen 3.5; without it `response` is empty |
| Temperature | 0.1 |
| num_predict | 256 |
| Retrieval budget | 4000 chars (500-char header + keyword-ranked paragraphs) |
| Validation split | 80/20 stratified by clause label, seed=42 → **40 val rows** |

### Text cleaning (`clean_text`)

- Normalize line endings
- Remove `QuickLinks -- ...` lines
- Remove lone page numbers / `-7-` markers
- Collapse 3+ newlines → 2; collapse repeated spaces
- **Does not** normalize `""` → `"` (gold spans use original quoting)

### Prompt template

```
Based ONLY on the contract text below, answer with the shortest possible exact quote from the text.

QUESTION: {full question string}

CONTRACT TEXT:
{retrieved excerpt — NOT full contract}

RULES:
- Return ONLY exact words/phrases copied from the contract
- One short phrase or sentence maximum when possible
- If not in the text, reply exactly: NOT FOUND

ANSWER:
```

### Span alignment

1. Strip prefixes / detect NOT FOUND → empty string
2. If prediction ∈ original context → use as-is
3. Else `rapidfuzz.partial_ratio_alignment` (cutoff 70)
4. Else sliding-window fuzzy match (cutoff 60)
5. Submission uses **aligned** span from **original** context

### Known bugs / limitations fixed during development

- **Empty predictions**: Qwen 3.5 default thinking mode + poisoned `cache.jsonl` with empty responses. Fixed with `think=False`, skip empty cache entries, `--refresh-cache`.
- **Alignment off-by-one**: e.g. `"roducer's..."` instead of `"Producer's..."` when fuzzy window misaligns.

### Project layout

- `run_qa_pipeline.py` — main pipeline
- `qa_pipeline.ipynb` — notebook wrapper
- `outputs/runs/<timestamp>_<model>_<mode>/` — per-run artifacts (`submission.csv`, `metrics.json`, `run_config.json`, prompt files)
- `outputs/latest_run.txt` — pointer to most recent run folder
- `outputs/cache.jsonl` — shared Ollama response cache

### Run command (latest full run)

```powershell
python run_qa_pipeline.py --mode full --model qwen3.5:0.8b --refresh-cache
```

---

## 5. Current results

### Latest full run

```text
Using Ollama model: qwen3.5:0.8b
Deleted stale cache.jsonl
Train: 200 rows, Test: 165 rows
Validation: 40/40 in ~19s (~2.08 it/s)
Validation exact match: 0.0250  (2.5%)
Test: 165/165 in ~76s (~2.16 it/s)
Saved submission.csv (165 rows)
```

### Saved metrics (`outputs/metrics.json`)

```json
{
  "val_exact_match": 0.025,
  "n_val": 40,
  "n_train": 160,
  "model": "qwen3.5:0.8b"
}
```

**→ 1 correct out of 40 validation examples** (exact character match).

### Error analysis (from `outputs/val_predictions.csv`)

| Failure mode | Example | Notes |
|--------------|---------|-------|
| **Wrong span, same topic** | License Grant: gold mentions "Producer further grants…"; pred is "Producer's Representations and Warranties…" | Retrieval/LLM finds related but wrong clause |
| **Wrong span entirely** | Notice Period: gold is 90-day renewal notice; pred is unrelated fee paragraph | Keyword retrieval misses correct paragraph |
| **Clause label as answer** | No-Solicit: pred = `No-Solicit Of Customers` | Model echoes question category, not contract text |
| **NOT FOUND** | Parties: gold = `"Nissin Holding")`; pred empty | Model fails on tricky entity extraction |
| **Almost correct** | Parties: gold = `Bravatek`; pred = `Bravatek and Fazync LLC, Bravatek and Fazync ag` | EM=0 due to extra text |
| **Negation flip** | Termination For Convenience: gold = "Either party may terminate…30 days"; pred = "Neither party may terminate without cause" | Semantic error |
| **Section header vs content** | Non-Transferable License: pred = `Section 1.1. License Grant.` | Too short / wrong granularity |
| **Correct** | Expiration Date id=78 | Full sentence match — **only confirmed EM hit** |

### Submission quality (`outputs/submission.csv`)

- **Not all empty** (earlier bug fixed), but many answers are empty, truncated, or wrong spans.
- Examples: id=0 → partial `"1.   Name of the Joint Ventur"`; id=2 → empty; id=1 → long but possibly wrong span.

---

## 6. Why results are poor (hypotheses for research)

### A. Generative LLM vs extractive EM metric

- Small LLM is asked to **quote** text but often **paraphrases, summarizes, or picks wrong clause**.
- Prompt says *"shortest possible exact quote"* / *"one short phrase"* — conflicts with gold answers that are **long verbatim clauses**.
- **Exact match** gives 0 for any extra/missing character, punctuation, or whitespace difference.

### B. Context truncation / retrieval

- Contracts are 5k–20k+ chars; only **~4000 chars** reach the model.
- Keyword paragraph retrieval is naive; correct clause often **not in excerpt** → NOT FOUND or wrong paragraph.
- `answer_start` in train shows answers appear anywhere in document, not just header.

### C. Train set too small for prompt-only LLM

- 200 examples, ~41 clause types → ~5 examples per type on average.
- Zero-shot / few-shot generative QA is weak; no fine-tuning on task format.

### D. Alignment pipeline issues

- Fuzzy alignment can return **wrong window** with high-ish fuzzy score.
- Cleaning for prompt but aligning to **original** text: model quotes cleaned version → alignment drift.
- Doubled quotes `""` in contracts may break substring match.

### E. Better-suited approaches (within rules)

1. **Fine-tuned extractive QA** (course seminar approach): `distilbert-base-cased-distilled-squad`, `deepset/bert-base-uncased-squad2` — predict start/end tokens, naturally outputs context substrings.
2. **Train on CUAD** (full dataset) + adapt to this competition format — allowed as extra data.
3. **Multi-pass retrieval**: BM25/dense retrieval over chunks, then extractive model or LLM on top-k chunks.
4. **Clause-specific heuristics**: regex/rules for Dates, Parties (first paragraph), Document Name (title line).
5. **Long-context handling**: sliding windows with overlap + vote/merge spans.
6. **Post-processing for EM**: trim to shortest matching span; try multiple candidate spans from LLM.
7. **Larger local model at 1B cap**: e.g. compare `qwen2.5:0.5b` vs `qwen3.5:0.8b` vs `llama3.2:1b` — still generative limitation remains.

---

## 7. Constraints checklist for proposed improvements

- [ ] Model ≤ **1B parameters**, runs **locally**
- [ ] No commercial **API** LLMs
- [ ] Output must be **exact substring** of test `context` (for EM)
- [ ] Can use **CUAD** and other public legal QA data for training
- [ ] Cannot use test labels for training
- [ ] Code must be **reproducible** and match submission approach

---

## 8. Files & artifacts for further analysis

| Path | Description |
|------|-------------|
| `data/train.csv`, `data/test.csv` | Raw data |
| `run_qa_pipeline.py` | Current pipeline |
| `outputs/val_predictions.csv` | 40 val rows with gold/pred/raw |
| `outputs/submission.csv` | Latest test submission |
| `outputs/metrics.json` | Val EM = 0.025 |
| `outputs/cache.jsonl` | Cached Ollama responses |
| `outputs/runs/` | Per-run folders with config, prompt, submission |
| `old_code/` | Earlier attempts (chat API fix, automodel BERT QA) |
| `Compition overview and rules.txt` | Official rules (Russian) |

---

## 9. Research questions for the next LLM

1. Should we **switch from generative LLM to fine-tuned extractive QA** given strict EM and small train set?
2. How to best **chunk long contracts** and aggregate span predictions?
3. Can we **mine CUAD** (510 contracts, 13k+ annotations) to pretrain/fine-tune within competition rules?
4. How to **normalize predictions** without breaking EM (if organizers use strict match)?
5. Which **clause types** fail most and need specialized handlers?
6. Is **paragraph keyword retrieval** sufficient, or do we need BM25/embeddings?
7. How to fix **answer boundary** selection (short entity vs full clause)?
8. What **validation strategy** is reliable with only 200 train samples?

---

## 10. Baseline comparison target

Course seminar (`rudn_26_seminar_10_qa.ipynb`) reports **~83% EM** on SQuAD validation with `distilbert-base-cased-distilled-squad` — far above our **2.5%** local validation. That suggests **extractive fine-tuning** is the most promising direction, adapted to legal contracts and this question format.

---

*Current best local validation: **2.5% exact match** (1/40) with `qwen3.5:0.8b` generative + retrieval pipeline.*
