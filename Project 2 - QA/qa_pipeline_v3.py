# -*- coding: utf-8 -*-
"""
Legal Contract QA Pipeline v3 (Refactored)
Features: Dynamic Model Selection, TTA Inference, Rank-Norm Ensemble, Validation Stage
"""

import gc
import json
import collections
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, DefaultDataCollator, TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

MODE = "full"  # "full", "val_only", "infer_only"
SKIP_TRAINING = True # Set to True to skip fine-tuning and just evaluate pretrained models

# Paths
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
OUTPUT_CSV = "./submission.csv"
VAL_METRICS_JSON = "./val_metrics.json"
VAL_PREDICTIONS_CSV = "./val_predictions.csv"

# Hyperparameters
SEED = 42
MAX_LENGTH = 384
MAX_ANS_LEN = 1000
N_BEST = 20
TTA_STRIDES = [64, 128, 192]  # Test-Time Augmentation strides
SWA_LAST_N = 2
VAL_SIZE = 0.2

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 0 if torch.cuda.is_available() else -1

# Model Candidates (Will auto-select up to 3-4 available ones)
CANDIDATE_MODELS = [
    "deepset/deberta-v3-base-squad2",
    "deepset/roberta-base-squad2",
    "akdeniz27/roberta-large-cuad",
    "mgigena/roberta-large-cuad",
]

# Weights for ensemble (higher = more trust)
MODEL_WEIGHTS = {
    "cuad": 4.0,
    "deberta": 3.5,
    "roberta-base": 3.0,
    "default": 2.5
}

# ==============================================================================
# SECTION 2: UTILITIES & DATA PREP
# ==============================================================================

def check_model_exists(model_name):
    """Quick check if model is downloadable."""
    try:
        AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return True
    except Exception:
        return False

def parse_clause_label(question: str) -> str:
    m = re.search(r'"([^"]+)"', str(question))
    return m.group(1).strip() if m else "Unknown"

def format_answers(row):
    """Convert flat CSV answers to SQuAD-style dict."""
    ans_text = str(row.get("answers", ""))
    start = int(row.get("answer_start", 0))
    if pd.isna(start) or start < 0:
        start = 0
    return {"text": [ans_text], "answer_start": [start]}

def clean_span(text: str) -> str:
    """Post-processing to fix minor punctuation/bracket mismatches."""
    if not text:
        return text
    text = text.strip()
    # Fix mismatched trailing brackets
    for o, c in [("(", ")"), ("[", "]"), ("{", "}")]:
        while text.endswith(o) and text.count(o) > text.count(c):
            text = text[:-1].strip()
        while text.startswith(c) and text.count(c) > text.count(o):
            text = text[1:].strip()
    # Strip leading/trailing commas and spaces
    while text and text[0] in ",; ":
        text = text[1:]
    while text and text[-1] in ",; ":
        text = text[:-1]
    return text.strip()

def load_and_split_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Format answers for HuggingFace
    train_df["answers"] = train_df.apply(format_answers, axis=1)
    train_df["clause_label"] = train_df["question"].apply(parse_clause_label)
    test_df["clause_label"] = test_df["question"].apply(parse_clause_label)
    
    # Stratified split
    rare = set(train_df["clause_label"].value_counts()[train_df["clause_label"].value_counts() < 2].index)
    stratify_col = train_df["clause_label"].apply(lambda x: "Other" if x in rare else x)
    
    tr_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, stratify=stratify_col, random_state=SEED
    )
    
    print(f"Data split: Train={len(tr_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return (
        Dataset.from_pandas(tr_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True))
    )

# ==============================================================================
# SECTION 3: TOKENIZATION
# ==============================================================================

def make_train_features(tokenizer):
    def _fn(examples):
        tok = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second", max_length=MAX_LENGTH,
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        offset_map = tok.pop("offset_mapping")
        starts, ends = [], []

        for i, offsets in enumerate(offset_map):
            input_ids = tok["input_ids"][i]
            seq_ids = tok.sequence_ids(i)
            answers = examples["answers"][sample_map[i]]

            if len(answers["answer_start"]) == 0 or not answers["text"][0]:
                starts.append(0) # CLS token
                ends.append(0)
                continue

            sc = answers["answer_start"][0]
            ec = sc + len(answers["text"][0])

            ts = next((j for j, s in enumerate(seq_ids) if s == 1), 0)
            te = len(seq_ids) - 1
            while te >= 0 and seq_ids[te] != 1:
                te -= 1

            if ts > te or not (offsets[ts][0] <= sc <= offsets[te][1]):
                starts.append(0)
                ends.append(0)
                continue

            s = ts
            while s < len(offsets) and offsets[s][0] <= sc:
                s += 1
            starts.append(s - 1)

            e = te
            while e >= ts and offsets[e][1] >= ec:
                e -= 1
            ends.append(e + 1)

        tok["start_positions"] = starts
        tok["end_positions"] = ends
        return tok
    return _fn

def make_val_features(tokenizer, stride):
    def _fn(examples):
        tok = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second", max_length=MAX_LENGTH,
            stride=stride, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            seq_ids = tok.sequence_ids(i)
            sample_idx = sample_map[i]
            tok["example_id"].append(examples["id"][sample_idx])
            # Mask offsets for question tokens (sequence_id == 0)
            tok["offset_mapping"][i] = [
                (o if seq_ids[k] == 1 else None)
                for k, o in enumerate(tok["offset_mapping"][i])
            ]
        return tok
    return _fn

# ==============================================================================
# SECTION 4: TRAINING & SWA
# ==============================================================================

class SWACallback(TrainerCallback):
    def __init__(self, last_n=2):
        self.last_n = last_n
        self.checkpoints = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.checkpoints.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if len(self.checkpoints) > self.last_n:
            self.checkpoints.pop(0)
        gc.collect()

def average_weights(model, checkpoints):
    if not checkpoints:
        return model
    avg = {}
    for key in checkpoints[0]:
        stacked = torch.stack([c[key].float() for c in checkpoints])
        avg[key] = stacked.mean(0).to(checkpoints[0][key].dtype)
    model.load_state_dict(avg)
    return model

def train_one_model(checkpoint, lr, epochs, batch, train_dataset):
    print(f"\n{'='*60}\nTRAINING: {checkpoint}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, use_safetensors=True)
    model.gradient_checkpointing_enable()

    tokenized_train = train_dataset.map(
        make_train_features(tok), batched=True,
        remove_columns=train_dataset.column_names,
    )

    swa_cb = SWACallback(last_n=SWA_LAST_N)
    args = TrainingArguments(
        output_dir=f"./tmp_ckpt_{checkpoint.replace('/','_')}",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.isAvailable(),
        logging_steps=50,
        save_strategy="no",
        seed=SEED,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized_train,
        data_collator=DefaultDataCollator(), callbacks=[swa_cb]
    )
    trainer.train()

    if len(swa_cb.checkpoints) >= 2:
        print("  -> Applying Stochastic Weight Averaging (SWA)...")
        model = average_weights(model, swa_cb.checkpoints)

    del trainer, tokenized_train, swa_cb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    return model, tok

# ==============================================================================
# SECTION 5: TTA INFERENCE
# ==============================================================================

def predict_with_tta(model, tokenizer, dataset, dataset_name="Inference"):
    print(f"\n  -> Running TTA Inference ({dataset_name}) with strides {TTA_STRIDES}...")
    device = next(model.parameters()).device
    id2idx = {k: i for i, k in enumerate(dataset["id"])}
    all_cands = collections.defaultdict(list)
    model.eval()

    for stride in TTA_STRIDES:
        print(f"     [TTA] stride={stride}...", end=" ", flush=True)
        tokenized = dataset.map(
            make_val_features(tokenizer, stride=stride),
            batched=True, remove_columns=dataset.column_names,
        )
        
        ids_all = tokenized["input_ids"]
        attn_all = tokenized["attention_mask"]
        all_sl, all_el = [], []

        with torch.no_grad():
            batch_size = 16
            for b in range(0, len(ids_all), batch_size):
                inp = torch.tensor(ids_all[b:b+batch_size]).to(device)
                attn = torch.tensor(attn_all[b:b+batch_size]).to(device)
                out = model(input_ids=inp, attention_mask=attn)
                all_sl.append(out.start_logits.cpu().numpy())
                all_el.append(out.end_logits.cpu().numpy())

        sl = np.concatenate(all_sl, 0)
        el = np.concatenate(all_el, 0)

        feat_per_ex = collections.defaultdict(list)
        for i, feat in enumerate(tokenized):
            feat_per_ex[id2idx[feat["example_id"]]].append(i)

        for ex_idx, example in enumerate(dataset):
            ctx = example["context"]
            for fi in feat_per_ex[ex_idx]:
                of = tokenized[fi]["offset_mapping"]
                sli = sl[fi]
                eli = el[fi]

                # Get top N start and end indices
                top_starts = np.argsort(sli)[-1:-N_BEST-1:-1].tolist()
                top_ends = np.argsort(eli)[-1:-N_BEST-1:-1].tolist()

                for si in top_starts:
                    for ei in top_ends:
                        if si >= len(of) or ei >= len(of):
                            continue
                        if of[si] is None or of[ei] is None:
                            continue
                        if ei < si or ei - si + 1 > MAX_ANS_LEN:
                            continue

                        text = clean_span(ctx[of[si][0]: of[ei][1]])
                        if not text:
                            continue

                        all_cands[example["id"]].append({
                            "score": float(sli[si]) + float(eli[ei]),
                            "text": text,
                        })
        
        del tokenized, ids_all, attn_all, all_sl, all_el, sl, el
        gc.collect()
        print("ok")

    # Aggregate best per example
    result = {}
    for qid in dataset["id"]:
        cands = all_cands.get(qid, [])
        if cands:
            best = max(cands, key=lambda x: x["score"])
            result[qid] = (best["text"], best["score"])
        else:
            result[qid] = ("", -1e9)
            
    del all_cands
    gc.collect()
    return result

# ==============================================================================
# SECTION 6: ENSEMBLE & EVALUATION
# ==============================================================================

def rank_normalize(all_preds):
    all_ids = list(all_preds[0].keys())
    normalized = []
    for preds in all_preds:
        scores = np.array([preds[qid][1] for qid in all_ids])
        ranks = np.argsort(np.argsort(scores)).astype(float)
        if ranks.max() > 0:
            ranks /= ranks.max()
        normalized.append({
            qid: (preds[qid][0], float(ranks[i]))
            for i, qid in enumerate(all_ids)
        })
    return normalized

def ensemble_final(all_preds, weights):
    normalized = rank_normalize(all_preds)
    final = {}
    all_ids = list(all_preds[0].keys())

    for qid in all_ids:
        scores = collections.defaultdict(float)
        votes = collections.Counter()

        for norm_preds, w in zip(normalized, weights):
            text, norm_score = norm_preds[qid]
            if not text.strip():
                continue
            scores[text] += w * norm_score
            votes[text] += 1

        # Consensus bonus
        for text, v in votes.items():
            if v >= 3:
                scores[text] *= 2.0
            elif v >= 2:
                scores[text] *= 1.5

        if scores:
            final[qid] = max(scores, key=scores.get)
        else:
            # Fallback to highest raw score
            best_text, best_score = "", -1e9
            for preds in all_preds:
                text, score = preds[qid]
                if score > best_score and text.strip():
                    best_text, best_score = text, score
            final[qid] = best_text

    return final

def evaluate(dataset, predictions, save_path):
    """Calculate Exact Match and save predictions."""
    df = dataset.to_pandas()
    df["prediction"] = df["id"].map(predictions).fillna("")
    df["clause_label"] = df["question"].apply(parse_clause_label)
    
    # Extract gold text from the dict
    df["gold"] = df["answers"].apply(lambda x: x["text"][0] if isinstance(x, dict) and x["text"] else "")
    df["em"] = (df["gold"].str.strip() == df["prediction"].str.strip()).astype(int)

    overall_em = df["em"].mean()
    clause_em = df.groupby("clause_label")["em"].agg(["mean", "count"]).rename(columns={"mean": "em", "count": "n"}).sort_values("em", ascending=False)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: Overall EM = {overall_em:.4f} ({int(df['em'].sum())}/{len(df)})")
    print(f"{'='*60}")
    print(clause_em.to_string())
    print(f"{'='*60}\n")

    df[["id", "question", "gold", "prediction", "clause_label", "em"]].to_csv(save_path, index=False)
    
    return {
        "val_exact_match": float(overall_em),
        "n_correct": int(df["em"].sum()),
        "n_val": len(df),
        "per_clause": clause_em["em"].to_dict()
    }

# ==============================================================================
# SECTION 7: MAIN ORCHESTRATOR
# ==============================================================================

def main():
    print(f"Device: {'GPU' if DEVICE == 0 else 'CPU'}")
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"GPU RAM Free: {free:.1f} GB")

    # 1. Select Models
    print("\nSelecting available models...")
    selected_models = []
    for m in CANDIDATE_MODELS:
        if check_model_exists(m):
            selected_models.append(m)
        if len(selected_models) >= 4: # Cap at 4 to save time/VRAM
            break
            
    if not selected_models:
        raise RuntimeError("No models could be loaded. Check internet connection.")
    
    # Build config for selected models
    models_config = []
    for m in selected_models:
        m_lower = m.lower()
        if "cuad" in m_lower:
            m_type, lr, epochs, batch = "cuad", 1e-5, 4, 4
        elif "large" in m_lower:
            m_type, lr, epochs, batch = "deberta", 1e-5, 4, 4
        else:
            m_type, lr, epochs, batch = "roberta-base", 2e-5, 3, 8
            
        models_config.append({
            "checkpoint": m,
            "lr": lr,
            "epochs": epochs,
            "batch": batch,
            "weight": MODEL_WEIGHTS.get(m_type, MODEL_WEIGHTS["default"])
        })

    # 2. Load Data
    train_ds, val_ds, test_ds = load_and_split_data()

    all_val_preds, all_test_preds = [], []
    weights = []

    # 3. Train & Infer Loop
    for i, cfg in enumerate(models_config):
        print(f"\n[{i+1}/{len(models_config)}] Processing: {cfg['checkpoint']}")
        
        if SKIP_TRAINING:
            print("  -> Skipping training (SKIP_TRAINING=True)")
            tok = AutoTokenizer.from_pretrained(cfg["checkpoint"], use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(cfg["checkpoint"], use_safetensors=True)
            
            # FORCE MODEL TO GPU
            device_str = "cuda" if DEVICE == 0 else "cpu"
            model = model.to(device_str)
            print(f"     -> Model loaded and moved to {device_str.upper()}")
            
            model.eval()
        else:
            model, tok = train_one_model(
                cfg["checkpoint"], cfg["lr"], cfg["epochs"], cfg["batch"], train_ds
            )

        # Validation Inference
        val_preds = predict_with_tta(model, tok, val_ds, dataset_name="Validation")
        all_val_preds.append(val_preds)
        
        # Test Inference (if not val_only)
        if MODE != "val_only":
            test_preds = predict_with_tta(model, tok, test_ds, dataset_name="Test")
            all_test_preds.append(test_preds)
            
        weights.append(cfg["weight"])
        
        # Cleanup
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Ensemble & Evaluate Validation
    print("\n" + "="*60)
    print("ENSEMBLING & EVALUATING")
    print("="*60)
    
    final_val_preds = ensemble_final(all_val_preds, weights) if len(all_val_preds) > 1 else all_val_preds[0]
    metrics = evaluate(val_ds, final_val_preds, VAL_PREDICTIONS_CSV)
    
    with open(VAL_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # 5. Generate Test Submission
    if MODE != "val_only":
        print("\nGenerating final submission...")
        final_test_preds = ensemble_final(all_test_preds, weights) if len(all_test_preds) > 1 else all_test_preds[0]
        
        sub_df = pd.read_csv(TEST_CSV)[["id"]].copy()
        sub_df["answers"] = sub_df["id"].map(final_test_preds).fillna("")
        sub_df.to_csv(OUTPUT_CSV, index=False)
        
        empty_count = (sub_df["answers"] == "").sum()
        print(f"Submission saved to: {OUTPUT_CSV}")
        print(f"Empty predictions: {empty_count}/{len(sub_df)}")

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()