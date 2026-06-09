# Requirements: pandas, torch, transformers, numpy, accelerate, sentencepiece, protobuf
# pip install pandas torch transformers numpy accelerate sentencepiece protobuf

import pandas as pd
import re
import ast
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    
# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================

def clean_text(text):
    """Cleans text by removing HTML tags, control characters, and normalizing whitespaces."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove weird control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_answer(ans):
    """Parses the answer from string representation of list or plain string."""
    if isinstance(ans, list):
        return str(ans[0]) if ans else ""
    if isinstance(ans, str):
        if ans.startswith('[') and ans.endswith(']'):
            try:
                parsed = ast.literal_eval(ans)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return str(parsed[0])
            except:
                pass
        return ans
    return str(ans)

def prepare_data(df):
    df['clean_context'] = df['context'].apply(clean_text)
    df['answer_text'] = df['answers'].apply(parse_answer).apply(clean_text)
    
    # Find answer start in cleaned context to align with token offsets
    starts = []
    valid_indices = []
    for i, row in df.iterrows():
        ctx = row['clean_context']
        ans = row['answer_text']
        start_idx = ctx.find(ans)
        if start_idx != -1 and ans:
            starts.append(start_idx)
            valid_indices.append(i)
        else:
            starts.append(-1)
            
    df['answer_start'] = starts
    # Keep only rows where we successfully found the answer in the cleaned context
    return df.iloc[valid_indices].reset_index(drop=True)

print("Loading and preprocessing data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df = prepare_data(train_df)
test_df['clean_context'] = test_df['context'].apply(clean_text)

# ==========================================
# 2. Tokenization and Dataset Preparation
# ==========================================

model_checkpoint = "deepset/bert-base-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_length = 512

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, is_test=False):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        context = row['clean_context']
        
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Robustly get sequence IDs (0 for question, 1 for context, None for special tokens)
        seq_ids = inputs.sequence_ids()
        # Replace None with 0 so we can convert to tensor
        seq_ids = [0 if s is None else s for s in seq_ids]
        inputs['sequence_ids'] = torch.tensor(seq_ids)
        
        if self.is_test:
            return {k: v.squeeze(0) for k, v in inputs.items()}
        
        answer_start = row['answer_start']
        answer_text = row['answer_text']
        
        offset_mapping = inputs['offset_mapping'].squeeze(0).tolist()
        
        start_positions = 0
        end_positions = 0
        
        if answer_start != -1:
            answer_end = answer_start + len(answer_text)
            token_start_index = 0
            token_end_index = 0
            
            # Find start token (only in context tokens where sequence_id == 1)
            for i, (start, end) in enumerate(offset_mapping):
                if seq_ids[i] == 1 and start <= answer_start < end:
                    token_start_index = i
                    break
            
            # Find end token
            for i, (start, end) in enumerate(offset_mapping):
                if seq_ids[i] == 1 and start < answer_end <= end:
                    token_end_index = i
                    break
            
            if token_start_index > 0 and token_end_index > 0 and token_start_index <= token_end_index:
                start_positions = token_start_index
                end_positions = token_end_index
                
        inputs['start_positions'] = torch.tensor(start_positions)
        inputs['end_positions'] = torch.tensor(end_positions)
        
        return {k: v.squeeze(0) for k, v in inputs.items()}

train_dataset = QADataset(train_df, tokenizer, max_length)
test_dataset = QADataset(test_df, tokenizer, max_length, is_test=True)

# ==========================================
# 3. Model Training
# ==========================================

print("Initializing model and starting training...")
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, use_safetensors=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    logging_steps=10,
    lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# ==========================================
# 4. Inference and Submission
# ==========================================

print("Running inference on test set...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
predictions = []

for i in range(len(test_dataset)):
    inputs = test_dataset[i]
    input_ids = inputs['input_ids'].unsqueeze(0).to(device)
    attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
    
    # Extract sequence_ids and offset_mapping
    seq_ids = inputs['sequence_ids'].numpy()
    offset_mapping = inputs['offset_mapping'].tolist()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # <--- FIXED: Cast to float32 to safely handle -1e9 masking
    start_logits = outputs.start_logits.cpu().float().numpy().squeeze()
    end_logits = outputs.end_logits.cpu().float().numpy().squeeze()
    
    # <--- FIXED: Mask out question and special tokens (where seq_ids != 1)
    start_logits[seq_ids != 1] = -1e9
    end_logits[seq_ids != 1] = -1e9
    
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)
    
    if start_idx > end_idx:
        end_idx = start_idx
        
    context = test_df.iloc[i]['clean_context']
    
    if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
        char_start = offset_mapping[start_idx][0]
        char_end = offset_mapping[end_idx][1]
        predicted_answer = context[char_start:char_end].strip()
    else:
        predicted_answer = ""
        
    predictions.append(predicted_answer)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'answers': predictions
})
submission.to_csv('submission.csv', index=False)
print("Success! Submission saved to submission.csv")