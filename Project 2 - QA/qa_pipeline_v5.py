# -*- coding: utf-8 -*-
"""
Legal Contract QA Pipeline v7 (Resumable Training + ALBERT Fix + Persistent Logging)
Features: Smart Checkpoint Resumption, Safe Gradient Checkpointing, Stage 1/2 Training, 
          TTA, Rank-Norm Ensemble, Test Set F1/EM, File Logging
"""

import gc
import json
import collections
import re
import string
import sys
import logging
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
# SECTION 1: LOGGING SETUP
# ==============================================================================

def setup_logger():
    logger = logging.getLogger("QAPipeline")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter('%(message)s'))

    # File Handler
    f_handler = logging.FileHandler('pipeline_run.log', mode='a', encoding='utf-8')
    f_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    # Suppress noisy logs but keep tqdm
    for lib in ["transformers", "datasets", "filelock", "huggingface_hub"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return logger

logger = setup_logger()

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

MODE = "full"  # "full", "val_only", "infer_only"
SKIP_TRAINING = False 

TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
CUAD_JSON_PATH = "data/cuad_data/train_separate_questions.json"
TEST_JSON_PATH = "data/cuad_data/test.json" 
OUTPUT_CSV = "submission.csv"
VAL_METRICS_JSON = "val_metrics.json"
VAL_PREDICTIONS_CSV = "val_predictions.csv"
SAVED_MODELS_DIR = "./saved_models"

SEED = 42
MAX_LENGTH = 512
MAX_ANS_LEN = 1000
N_BEST = 50
TTA_STRIDES = [64, 128, 192]
SWA_LAST_N = 2
VAL_SIZE = 0.2
DOMAIN_ADAPT_SIZE = 2500 
APPLY_CLEAN_SPAN = True  

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 0 if torch.cuda.is_available() else -1

CANDIDATE_MODELS = [
    "akdeniz27/roberta-large-cuad", "mgigena/roberta-large-cuad",
    "Narrativa/roberta-large-cuad", "Yanzhu/roberta-large-cuad",
    "deepset/deberta-v3-base-squad2", "deepset/roberta-base-squad2",
    "deepset/deberta-v3-large-squad2", "deepset/electra-base-squad2",
    "twmkn9/albert-base-v2-squad2", "mrm8488/bert-small-finetuned-squadv2",
]

MODEL_WEIGHTS = {
    "cuad": 5.0, "deberta-large": 3.5, "deberta-base": 3.0, 
    "roberta-large": 2.5, "roberta-base": 2.0, "electra-base": 1.8, 
    "albert-base": 1.5, "bert-small": 1.0, "default": 1.5
}

# ==============================================================================
# SECTION 3: METRICS & UTILITIES
# ==============================================================================

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_exact(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def get_model_config(model_name):
    m_lower = model_name.lower()
    if "cuad" in m_lower: model_type = "cuad"
    elif "deberta" in m_lower: model_type = "deberta-large" if "large" in m_lower else "deberta-base"
    elif "roberta" in m_lower: model_type = "roberta-large" if "large" in m_lower else "roberta-base"
    elif "electra" in m_lower: model_type = "electra-base"
    elif "albert" in m_lower: model_type = "albert-base"
    elif "bert" in m_lower and "small" in m_lower: model_type = "bert-small"
    else: model_type = "default"
    
    if "cuad" in m_lower: lr, epochs, batch = 1e-5, 5, 256
    elif "large" in m_lower or "albert" in m_lower:
        lr, epochs, batch = (1e-5, 3, 128) if "albert" in m_lower else (1e-5, 4, 64)
    elif "deberta" in m_lower: lr, epochs, batch = 2e-5, 4, 512
    elif "electra" in m_lower: lr, epochs, batch = 3e-5, 3, 512
    elif "bert" in m_lower and "small" in m_lower: lr, epochs, batch = 3e-5, 3, 512
    else: lr, epochs, batch = 2e-5, 3, 256
    
    return {"checkpoint": model_name, "lr": lr, "epochs": epochs, "batch": batch,
            "weight": MODEL_WEIGHTS.get(model_type, 1.5), "type": model_type}

def select_best_models(max_models=8):
    if torch.cuda.is_available():
        total_ram = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_gpu_ram_gb = total_ram - 2
    else:
        available_gpu_ram_gb = 0
        
    available = [m for m in CANDIDATE_MODELS if check_model_exists(m)]
    if not available: raise RuntimeError("No models available!")
    
    cuad_models = [m for m in available if "cuad" in m.lower()][:3]
    general_models = [m for m in available if "cuad" not in m.lower() and ("large" in m.lower() or "deberta" in m.lower())]
    selected = cuad_models + general_models[:max_models - len(cuad_models)]
    
    if len(selected) < max_models:
        remaining = [m for m in available if m not in selected]
        selected.extend(remaining[:max_models - len(selected)])
        
    return selected[:max_models]

def check_model_exists(model_name):
    try:
        AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return True
    except Exception: return False

def parse_clause_label(question: str) -> str:
    m = re.search(r'"([^"]+)"', str(question))
    return m.group(1).strip() if m else "Unknown"

def format_answers(row):
    ans_text = str(row.get("answers", ""))
    start = int(row.get("answer_start", 0))
    if pd.isna(start) or start < 0: start = 0
    return {"text": [ans_text], "answer_start": [start]}

def clean_span(text: str) -> str:
    if not text: return text
    text = text.strip()
    for o, c in [("(", ")"), ("[", "]"), ("{", "}")]:
        while text.endswith(o) and text.count(o) > text.count(c): text = text[:-1].strip()
        while text.startswith(c) and text.count(c) > text.count(o): text = text[1:].strip()
    while text and text[0] in ",; ": text = text[1:]
    while text and text[-1] in ",; ": text = text[:-1]
    return text.strip()

def get_save_dir(checkpoint):
    """Get standardized save directory for a model."""
    safe_name = checkpoint.replace('/', '_').replace('-', '_')
    return Path(SAVED_MODELS_DIR) / safe_name

# ==============================================================================
# SECTION 4: DATA LOADING & STAGE 1 PREP
# ==============================================================================

def parse_cuad_json_to_dataset(json_path, max_samples=2500):
    logger.info(f"\nLoading CUAD JSON for Stage 1 Domain Adaptation...")
    with open(json_path, 'r', encoding='utf-8') as f:
        cuad_raw = json.load(f)
    
    examples = []
    for doc in cuad_raw['data']:
        for para in doc['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                answers = qa.get('answers', [])
                if not answers: continue
                examples.append({
                    "id": str(qa['id']),
                    "question": qa['question'],
                    "context": context,
                    "answers": {"text": [a['text'] for a in answers], "answer_start": [a['answer_start'] for a in answers]},
                    "clause_label": parse_clause_label(qa['question'])
                })
    
    df = pd.DataFrame(examples)
    n_classes = df['clause_label'].nunique()
    target_per_class = max(1, max_samples // n_classes)
    
    sampled_dfs = []
    for clause, group in df.groupby('clause_label'):
        if len(group) <= target_per_class: 
            sampled_dfs.append(group)
        else: 
            sampled_dfs.append(group.sample(n=target_per_class, random_state=SEED))
            
    df_sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    logger.info(f"✅ Stratified Stage 1 sample: {len(df_sampled)} examples across {n_classes} classes.")
    return Dataset.from_pandas(df_sampled)

def load_and_split_data():
    logger.info("Loading Stage 2 Data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    train_df["answers"] = train_df.apply(format_answers, axis=1)
    train_df["clause_label"] = train_df["question"].apply(parse_clause_label)
    test_df["clause_label"] = test_df["question"].apply(parse_clause_label)
    
    rare = set(train_df["clause_label"].value_counts()[train_df["clause_label"].value_counts() < 2].index)
    stratify_col = train_df["clause_label"].apply(lambda x: "Other" if x in rare else x)
    
    tr_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, stratify=stratify_col, random_state=SEED)
    
    return (
        Dataset.from_pandas(tr_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True))
    )

# ==============================================================================
# SECTION 5: TOKENIZATION & TRAINING
# ==============================================================================

def make_train_features(tokenizer):
    def _fn(examples):
        tok = tokenizer(examples["question"], examples["context"], truncation="only_second", 
                        max_length=MAX_LENGTH, stride=128, return_overflowing_tokens=True,
                        return_offsets_mapping=True, padding="max_length")
        sample_map = tok.pop("overflow_to_sample_mapping")
        offset_map = tok.pop("offset_mapping")
        starts, ends = [], []

        for i, offsets in enumerate(offset_map):
            seq_ids = tok.sequence_ids(i)
            answers = examples["answers"][sample_map[i]]
            if len(answers["answer_start"]) == 0 or not answers["text"][0]:
                starts.append(0); ends.append(0); continue

            sc = answers["answer_start"][0]
            ec = sc + len(answers["text"][0])
            ts = next((j for j, s in enumerate(seq_ids) if s == 1), 0)
            te = len(seq_ids) - 1
            while te >= 0 and seq_ids[te] != 1: te -= 1

            if ts > te or not (offsets[ts][0] <= sc <= offsets[te][1]):
                starts.append(0); ends.append(0); continue

            s = ts
            while s < len(offsets) and offsets[s][0] <= sc: s += 1
            starts.append(s - 1)
            e = te
            while e >= ts and offsets[e][1] >= ec: e -= 1
            ends.append(e + 1)

        tok["start_positions"] = starts
        tok["end_positions"] = ends
        return tok
    return _fn

def make_val_features(tokenizer, stride):
    def _fn(examples):
        tok = tokenizer(examples["question"], examples["context"], truncation="only_second", 
                        max_length=MAX_LENGTH, stride=stride, return_overflowing_tokens=True,
                        return_offsets_mapping=True, padding="max_length")
        sample_map = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            seq_ids = tok.sequence_ids(i)
            tok["example_id"].append(examples["id"][sample_map[i]])
            tok["offset_mapping"][i] = [(o if seq_ids[k] == 1 else None) for k, o in enumerate(tok["offset_mapping"][i])]
        return tok
    return _fn

class SWACallback(TrainerCallback):
    def __init__(self, last_n=2):
        self.last_n = last_n
        self.checkpoints = []
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.checkpoints.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if len(self.checkpoints) > self.last_n: self.checkpoints.pop(0)

def average_weights(model, checkpoints):
    avg = {}
    for key in checkpoints[0]:
        stacked = torch.stack([c[key].float() for c in checkpoints])
        avg[key] = stacked.mean(0).to(checkpoints[0][key].dtype)
    model.load_state_dict(avg)
    return model

def train_one_model(checkpoint, lr, epochs, batch, stage2_dataset, stage1_dataset=None):
    save_dir = get_save_dir(checkpoint)
    
    # SMART RESUMPTION: Check if already trained
    if save_dir.exists() and (save_dir / "config.json").exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ FOUND SAVED CHECKPOINT: {checkpoint}")
        logger.info(f"   Loading from {save_dir} (skipping training)")
        logger.info(f"{'='*60}")
        
        tok = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
        # Fallback to hub tokenizer if local save is incomplete
        if tok.vocab_size == 0:
            tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            
        model = AutoModelForQuestionAnswering.from_pretrained(save_dir)
        device_str = "cuda" if DEVICE == 0 else "cpu"
        model = model.to(device_str)
        model.eval()
        return model, tok
    
    logger.info(f"\n{'='*60}\nTRAINING: {checkpoint}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, use_safetensors=True)
    
    # SAFE GRADIENT CHECKPOINTING (Fixes ALBERT crash)
    try:
        model.gradient_checkpointing_enable()
        logger.info("  ✅ Gradient checkpointing enabled.")
    except ValueError:
        logger.info(f"  ⚠️ Gradient checkpointing not supported for {checkpoint}, skipping.")

    # STAGE 1: Domain Adaptation
    if stage1_dataset is not None:
        logger.info(f"  -> STAGE 1: Domain Adaptation on {len(stage1_dataset)} CUAD examples...")
        tok_stage1 = stage1_dataset.map(make_train_features(tok), batched=True, remove_columns=stage1_dataset.column_names)
        args_stage1 = TrainingArguments(
            output_dir=f"./tmp_stage1_{checkpoint.replace('/','_')}", learning_rate=lr,
            per_device_train_batch_size=batch, gradient_accumulation_steps=4, num_train_epochs=2,
            weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type="cosine", fp16=torch.cuda.is_available(),
            logging_steps=50, save_strategy="no", seed=SEED, report_to="none"
        )
        trainer_s1 = Trainer(model=model, args=args_stage1, train_dataset=tok_stage1, data_collator=DefaultDataCollator())
        trainer_s1.train()
        del trainer_s1, tok_stage1
        gc.collect(); torch.cuda.empty_cache()

    # STAGE 2: Calibration
    logger.info(f"  -> STAGE 2: Calibration on {len(stage2_dataset)} specific examples...")
    tokenized_train = stage2_dataset.map(make_train_features(tok), batched=True, remove_columns=stage2_dataset.column_names)
    swa_cb = SWACallback(last_n=SWA_LAST_N)
    args = TrainingArguments(
        output_dir=f"./tmp_ckpt_{checkpoint.replace('/','_')}", learning_rate=lr,
        per_device_train_batch_size=batch, gradient_accumulation_steps=4, num_train_epochs=epochs,
        weight_decay=0.05, warmup_ratio=0.15, lr_scheduler_type="cosine_with_restarts",
        fp16=torch.cuda.is_available(), logging_steps=10, save_strategy="no", seed=SEED, report_to="none"
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, data_collator=DefaultDataCollator(), callbacks=[swa_cb])
    trainer.train()

    # Apply SWA
    if len(swa_cb.checkpoints) >= 2:
        logger.info("  -> Applying Stochastic Weight Averaging (SWA)...")
        model = average_weights(model, swa_cb.checkpoints)

    # SAVE FINAL MODEL TO DISK
    logger.info(f"  💾 Saving final model to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    logger.info(f"  ✅ Model saved successfully.")

    del trainer, tokenized_train, swa_cb
    gc.collect(); torch.cuda.empty_cache()
    
    device_str = "cuda" if DEVICE == 0 else "cpu"
    model = model.to(device_str)
    model.eval()
    return model, tok

# ==============================================================================
# SECTION 6: TTA INFERENCE
# ==============================================================================

def predict_with_tta(model, tokenizer, dataset, dataset_name="Inference"):
    logger.info(f"\n  -> Running TTA Inference ({dataset_name})...")
    device = next(model.parameters()).device
    id2idx = {k: i for i, k in enumerate(dataset["id"])}
    all_cands = collections.defaultdict(list)

    for stride in TTA_STRIDES:
        tokenized = dataset.map(make_val_features(tokenizer, stride=stride), batched=True, remove_columns=dataset.column_names)
        ids_all = tokenized["input_ids"]
        attn_all = tokenized["attention_mask"]
        all_sl, all_el = [], []

        with torch.no_grad():
            for b in range(0, len(ids_all), 16):
                inp = torch.tensor(ids_all[b:b+16]).to(device)
                attn = torch.tensor(attn_all[b:b+16]).to(device)
                out = model(input_ids=inp, attention_mask=attn)
                all_sl.append(out.start_logits.cpu().numpy())
                all_el.append(out.end_logits.cpu().numpy())

        sl = np.concatenate(all_sl, 0)
        el = np.concatenate(all_el, 0)
        feat_per_ex = collections.defaultdict(list)
        for i, feat in enumerate(tokenized): feat_per_ex[id2idx[feat["example_id"]]].append(i)

        for ex_idx, example in enumerate(dataset):
            ctx = example["context"]
            for fi in feat_per_ex[ex_idx]:
                of = tokenized[fi]["offset_mapping"]
                sli, eli = sl[fi], el[fi]
                top_starts = np.argsort(sli)[-1:-N_BEST-1:-1].tolist()
                top_ends = np.argsort(eli)[-1:-N_BEST-1:-1].tolist()

                for si in top_starts:
                    for ei in top_ends:
                        if si >= len(of) or ei >= len(of) or of[si] is None or of[ei] is None: continue
                        if ei < si or ei - si + 1 > MAX_ANS_LEN: continue
                        
                        text = ctx[of[si][0]: of[ei][1]]
                        if APPLY_CLEAN_SPAN: text = clean_span(text)
                        if not text: continue

                        all_cands[example["id"]].append({"score": float(sli[si]) + float(eli[ei]), "text": text})
        del tokenized, sl, el
        gc.collect()

    result = {}
    for qid in dataset["id"]:
        cands = all_cands.get(qid, [])
        if cands:
            best = max(cands, key=lambda x: x["score"])
            result[qid] = (best["text"], best["score"])
        else:
            result[qid] = ("", -1e9)
    return result

# ==============================================================================
# SECTION 7: ENSEMBLE & EVALUATION
# ==============================================================================

def rank_normalize(all_preds):
    all_ids = list(all_preds[0].keys())
    normalized = []
    for preds in all_preds:
        scores = np.array([preds[qid][1] for qid in all_ids])
        ranks = np.argsort(np.argsort(scores)).astype(float)
        if ranks.max() > 0: ranks /= ranks.max()
        normalized.append({qid: (preds[qid][0], float(ranks[i])) for i, qid in enumerate(all_ids)})
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
            if not text.strip(): continue
            scores[text] += w * norm_score
            votes[text] += 1
        for text, v in votes.items():
            if v >= 3: scores[text] *= 2.0
            elif v >= 2: scores[text] *= 1.5
        if scores: final[qid] = max(scores, key=scores.get)
        else: final[qid] = max([preds[qid][0] for preds in all_preds if preds[qid][0].strip()], default="")
    return final

def evaluate(dataset, predictions, save_path):
    df = dataset.to_pandas()
    df["prediction"] = df["id"].map(predictions).fillna("")
    df["clause_label"] = df["question"].apply(parse_clause_label)
    df["gold"] = df["answers"].apply(lambda x: x["text"][0] if isinstance(x, dict) and x["text"] else "")
    
    df["em"] = (df["gold"].str.strip() == df["prediction"].str.strip()).astype(int)
    df["f1"] = df.apply(lambda row: compute_f1(row["prediction"], row["gold"]), axis=1)

    overall_em = df["em"].mean()
    overall_f1 = df["f1"].mean()
    
    logger.info(f"\nVALIDATION RESULTS: Overall EM = {overall_em:.4f} | Overall F1 = {overall_f1:.4f}")
    df[["id", "question", "gold", "prediction", "clause_label", "em", "f1"]].to_csv(save_path, index=False)
    return {"val_exact_match": float(overall_em), "val_f1": float(overall_f1)}

def evaluate_test_set(submission_csv, test_csv, test_json_path):
    logger.info("\n" + "="*60)
    logger.info("EVALUATING HIDDEN TEST SET")
    logger.info("="*60)
    
    pred_df = pd.read_csv(submission_csv)
    test_df = pd.read_csv(test_csv)
    merged_df = pred_df.merge(test_df[['id', 'question', 'context']], on='id', how='left')
    
    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
    except Exception as e:
        logger.error(f"Could not load test JSON: {e}")
        return

    gt_by_id = {}
    gt_by_content = {}
    
    if 'data' in gt_data:
        for doc in gt_data['data']:
            for para in doc['paragraphs']:
                context = para['context']
                ctx_snippet = normalize_answer(context)[:100]
                for qa in para['qas']:
                    qid = str(qa['id'])
                    q_text = normalize_answer(qa['question'])
                    answers = [ans['text'] for ans in qa.get('answers', [])]
                    gt_by_id[qid] = answers
                    gt_by_content[(q_text, ctx_snippet)] = answers
                    
    total_em, total_f1, matched = 0, 0, 0
    
    for idx, row in merged_df.iterrows():
        csv_id = str(row['id'])
        pred_text = str(row['answers'])
        q_text = normalize_answer(row['question'])
        ctx_snippet = normalize_answer(str(row['context']))[:100]
        
        gold_answers = gt_by_id.get(csv_id) or gt_by_content.get((q_text, ctx_snippet))
        
        if gold_answers is not None:
            matched += 1
            if not gold_answers:
                if pred_text == "": total_em += 1; total_f1 += 1
                continue
            max_em = max(compute_exact(pred_text, ga) for ga in gold_answers)
            max_f1 = max(compute_f1(pred_text, ga) for ga in gold_answers)
            total_em += max_em
            total_f1 += max_f1
            
    if matched > 0:
        logger.info(f"Matched {matched} predictions to ground truth.")
        logger.info(f"🏆 TEST SET EXACT MATCH (EM): {total_em / matched * 100:.2f}%")
        logger.info(f"🏆 TEST SET F1 SCORE:         {total_f1 / matched * 100:.2f}%")
    else:
        logger.warning("❌ No matches found for test set evaluation.")

# ==============================================================================
# SECTION 8: MAIN ORCHESTRATOR
# ==============================================================================

def main():
    logger.info(f"Device: {'GPU' if DEVICE == 0 else 'CPU'}")
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        logger.info(f"GPU RAM Free: {free:.1f} GB")

    selected_models = select_best_models(max_models=8)
    models_config = [get_model_config(m) for m in selected_models]
    
    train_ds, val_ds, test_ds = load_and_split_data()
    
    stage1_ds = None
    if not SKIP_TRAINING:
        stage1_ds = parse_cuad_json_to_dataset(CUAD_JSON_PATH, max_samples=DOMAIN_ADAPT_SIZE)

    all_val_preds, all_test_preds, weights = [], [], []

    for i, cfg in enumerate(models_config):
        logger.info(f"\n[{i+1}/{len(models_config)}] Processing: {cfg['checkpoint']}")
        
        if SKIP_TRAINING:
            tok = AutoTokenizer.from_pretrained(cfg["checkpoint"], use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(cfg["checkpoint"], use_safetensors=True)
            device_str = "cuda" if DEVICE == 0 else "cpu"
            model = model.to(device_str)
            model.eval()
        else:
            needs_stage1 = ("cuad" not in cfg["checkpoint"].lower()) and (stage1_ds is not None)
            model, tok = train_one_model(
                cfg["checkpoint"], cfg["lr"], cfg["epochs"], cfg["batch"], 
                stage2_dataset=train_ds, 
                stage1_dataset=stage1_ds if needs_stage1 else None
            )

        all_val_preds.append(predict_with_tta(model, tok, val_ds, "Validation"))
        if MODE != "val_only":
            all_test_preds.append(predict_with_tta(model, tok, test_ds, "Test"))
        weights.append(cfg["weight"])
        
        del model, tok
        gc.collect(); torch.cuda.empty_cache()

    if MODE != "val_only" and MODE != "infer_only":
        final_val_preds = ensemble_final(all_val_preds, weights) if len(all_val_preds) > 1 else all_val_preds[0]
        metrics = evaluate(val_ds, final_val_preds, VAL_PREDICTIONS_CSV)
        with open(VAL_METRICS_JSON, "w") as f: json.dump(metrics, f, indent=2)

    if MODE != "val_only":
        final_test_preds = ensemble_final(all_test_preds, weights) if len(all_test_preds) > 1 else all_test_preds[0]
        sub_df = pd.read_csv(TEST_CSV)[["id"]].copy()
        sub_df["answers"] = sub_df["id"].map(final_test_preds).fillna("")
        sub_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"\nSubmission saved to: {OUTPUT_CSV}")
        
        evaluate_test_set(OUTPUT_CSV, TEST_CSV, TEST_JSON_PATH)

    logger.info("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()