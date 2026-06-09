#!/usr/bin/env python
# coding: utf-8

# ## 1. Imports and Setup

# In[1]:


# Cell 1: Imports and configuration
import pandas as pd
import re
import json
from pathlib import Path
from typing import Optional, List, Dict
import ollama
import time

# Configuration
DATA_DIR = Path("data")
MODEL_NAME = "qwen3.5:0.8b"  # Adjust to your downloaded model name
MAX_CONTEXT_LENGTH = 4000  # Truncate very long contexts
TEMPERATURE = 0.1  # Low temperature for more deterministic answers
TIMEOUT_SECONDS = 60  # Timeout for LLM requests

# Set pandas options for better display
pd.set_option('display.max_colwidth', None)


# ## 2. Text Cleaning Functions

# In[2]:


# Cell 2: Text preprocessing functions
def clean_context(text: str) -> str:
    """
    Clean contract text by removing HTML tags, excessive whitespace, 
    and other artifacts while preserving meaningful content.

    Args:
        text: Raw context text from the dataset

    Returns:
        Cleaned text suitable for LLM processing
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown-style links but keep the text: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove excessive whitespace (multiple spaces/tabs/newlines)
    text = re.sub(r'\s+', ' ', text)

    # Remove control characters except basic whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize quotes and dashes
    text = text.replace('""', '"').replace("''", "'")
    text = text.replace('—', '-').replace('–', '-')

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def truncate_context(text: str, max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """
    Truncate context to fit within model's context window while 
    trying to preserve the most relevant content.

    Args:
        text: Cleaned context text
        max_length: Maximum number of characters to keep

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Keep the beginning and end, as contract answers often appear in definitions or clauses
    # Keep ~60% from start, ~40% from end
    start_len = int(max_length * 0.6)
    end_len = max_length - start_len

    return text[:start_len] + " ... [truncated] ... " + text[-end_len:]


# ## 3. Data Loading and Preparation

# In[3]:


# Cell 3: Load and prepare datasets
def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def prepare_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply preprocessing to dataframe: clean context, parse answers.

    Args:
        df: Input dataframe
        is_train: Whether this is training data (has 'answers' column)

    Returns:
        Preprocessed dataframe
    """
    df = df.copy()

    # Clean context column
    df['context_clean'] = df['context'].apply(clean_context)
    df['context_clean'] = df['context_clean'].apply(truncate_context)

    # Parse answers column (it's stored as a string representation of a list)
    if is_train and 'answers' in df.columns:
        def parse_answers(ans_str: str) -> str:
            """Extract the primary answer from the answers field."""
            if pd.isna(ans_str) or ans_str == '':
                return ""
            try:
                # Handle string representation of list
                if isinstance(ans_str, str) and ans_str.startswith('['):
                    answers_list = json.loads(ans_str.replace("'", '"'))
                    return answers_list[0] if answers_list else ""
                return str(ans_str)
            except:
                return str(ans_str)

        df['answer_target'] = df['answers'].apply(parse_answers)

    return df


# ## 4. LLM Prompt Engineering and Inference

# In[10]:


# Cell 4: LLM inference with improved prompt
def build_prompt(question: str, context: str) -> str:
    """
    Construct a focused prompt for extractive QA.
    """
    prompt = f"""Based ONLY on the contract text below, answer this question with the shortest possible exact answer.

QUESTION: {question}

CONTRACT TEXT:
{context}

IMPORTANT RULES:
- Answer with ONLY the exact words/phrase from the contract
- Use ONE short phrase (maximum 10 words)
- For dates, use format like "January 1, 2000"
- For party names, copy exactly as written
- If the exact answer is NOT in the text, respond with "NOT FOUND"

ANSWER:"""

    return prompt


def get_llm_answer(question: str, context: str, model: str = MODEL_NAME) -> Optional[str]:
    """Query the local LLM for an answer with better error handling."""
    prompt = build_prompt(question, context)

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 50,  # Short answers only
                "top_k": 10,
                "top_p": 0.5,
            }
        )

        answer = response['response'].strip()

        # Clean up common prefixes
        answer = re.sub(r'^(ANSWER|Answer|answer):\s*', '', answer)
        answer = re.sub(r'^["\']|["\']$', '', answer)

        # Check for "not found" responses
        if any(phrase in answer.lower() for phrase in ['not found', 'cannot find', "doesn't contain", "does not contain"]):
            return ""

        return answer if answer else ""

    except Exception as e:
        print(f"Error: {e}")
        return ""


def predict_row(row: pd.Series, idx: int, total: int) -> str:
    """Predict for a single row."""
    print(f"  [{idx+1}/{total}] Q: {row['question'][:60]}...", end=" ")

    answer = get_llm_answer(row['question'], row['context_clean'])

    if answer:
        print(f"-> {answer[:50]}")
    else:
        print("-> [NOT FOUND]")

    return answer


# In[14]:


# Cell 5: Fixed batch prediction
def batch_predict(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Generate predictions with proper progress tracking.
    """
    df = df.copy()
    predictions = []
    total = len(df)

    print(f"\nGenerating predictions for {total} examples...")

    for idx, row in df.iterrows():
        if verbose:
            print(f"\n[{idx+1}/{total}]")
            print(f"  Question: {row['question'][:100]}...")

        answer = get_llm_answer(row['question'], row['context_clean'])

        if answer:
            predictions.append(answer)
            if verbose:
                print(f"  Answer: {answer[:80]}")
        else:
            predictions.append("")
            if verbose:
                print(f"  Answer: [NOT FOUND in contract]")

        # Small delay between requests
        if (idx + 1) % 5 == 0:
            time.sleep(0.5)

    df['prediction'] = predictions
    return df


def predict_sample(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Predict on a small sample for testing."""
    sample = df.head(n).copy()
    sample['prediction'] = None

    for idx, row in sample.iterrows():
        answer = get_llm_answer(row['question'], row['context_clean'])
        sample.at[idx, 'prediction'] = answer if answer else ""
        time.sleep(0.5)

    return sample


# ## 5. Evaluation Function (for validation)

# In[15]:


# Cell 6: Evaluation functions
def exact_match(pred: str, true: str) -> bool:
    """Check if two answers match exactly (case-insensitive, normalized)."""
    if pd.isna(pred) or pd.isna(true):
        return False

    pred_norm = str(pred).strip().lower()
    true_norm = str(true).strip().lower()

    # Remove punctuation for comparison
    pred_norm = re.sub(r'[^\w\s]', '', pred_norm)
    true_norm = re.sub(r'[^\w\s]', '', true_norm)

    # Remove extra spaces
    pred_norm = re.sub(r'\s+', ' ', pred_norm)
    true_norm = re.sub(r'\s+', ' ', true_norm)

    return pred_norm == true_norm


def calculate_exact_match_score(df: pd.DataFrame) -> float:
    """Calculate exact match score for the dataframe."""
    if 'prediction' not in df.columns or 'answer_target' not in df.columns:
        print("Missing required columns: 'prediction' or 'answer_target'")
        return 0.0

    matches = 0
    total = len(df)

    for idx, row in df.iterrows():
        if exact_match(row['prediction'], row['answer_target']):
            matches += 1

    score = matches / total if total > 0 else 0.0
    print(f"Exact Match: {matches}/{total} = {score:.3f}")

    return score


def show_errors(df: pd.DataFrame, n: int = 10):
    """Show examples where predictions don't match."""
    errors = []

    for idx, row in df.iterrows():
        if not exact_match(row.get('prediction', ''), row.get('answer_target', '')):
            errors.append({
                'question': row['question'][:100],
                'true_answer': row['answer_target'],
                'predicted': row.get('prediction', ''),
                'context_preview': row['context_clean'][:200]
            })

    errors_df = pd.DataFrame(errors)
    print(f"\n{len(errors_df)} errors found. Showing first {min(n, len(errors_df))}:")
    print(errors_df.head(n).to_string())

    return errors_df


# In[16]:


# Cell 7: Test on training examples to verify the model works
def test_on_training_examples(train_df: pd.DataFrame, n: int = 5):
    """Test the pipeline on a few training examples."""
    print("=" * 60)
    print("TESTING ON TRAINING EXAMPLES")
    print("=" * 60)

    sample = train_df.head(n).copy()
    sample['prediction'] = None

    for idx, row in sample.iterrows():
        print(f"\n--- Example {idx+1} ---")
        print(f"Question: {row['question']}")
        print(f"True Answer: {row['answer_target']}")

        answer = get_llm_answer(row['question'], row['context_clean'])
        sample.at[idx, 'prediction'] = answer if answer else ""

        print(f"Predicted: {sample.at[idx, 'prediction']}")
        print(f"Match: {'✓' if exact_match(answer, row['answer_target']) else '✗'}")

        time.sleep(0.5)

    score = calculate_exact_match_score(sample)
    return sample, score

# Load and prepare data
train_df, test_df = load_data()
train_df = prepare_dataframe(train_df, is_train=True)
test_df = prepare_dataframe(test_df, is_train=False)

# Test on small sample
sample_results, sample_score = test_on_training_examples(train_df, n=5)


# In[18]:


# Cell 8: Check what models are available
def list_ollama_models():
    """List available models in Ollama."""
    try:
        response = ollama.list()
        models = response['models']
        print("Available Ollama models:")
        for model in models:
            print(f"  - {model['name']}")
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

# Uncomment to check available models
available_models = list_ollama_models()


# In[19]:


# Cell 1: Proper Ollama configuration and model discovery
import pandas as pd
import re
import json
from pathlib import Path
from typing import Optional
import subprocess
import sys

DATA_DIR = Path("data")

def get_available_ollama_models():
    """Get list of available models from Ollama with proper parsing."""
    try:
        # Use subprocess to get models
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    # Format: "qwen:0.5b                4.2 GB · 9a5f3f6cba5c"
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def test_ollama_connection(model_name: str) -> bool:
    """Test if a specific model works."""
    try:
        import ollama
        response = ollama.generate(
            model=model_name,
            prompt="Say 'OK'",
            options={"num_predict": 5}
        )
        print(f"✓ Model '{model_name}' is working!")
        print(f"  Response: {response['response'][:50]}")
        return True
    except Exception as e:
        print(f"✗ Model '{model_name}' failed: {e}")
        return False

# Check available models
print("=" * 60)
print("CHECKING OLLAMA SETUP")
print("=" * 60)

available_models = get_available_ollama_models()

if available_models:
    print(f"\nAvailable models: {available_models}")

    # Try each model until one works
    MODEL_NAME = None
    for model in available_models:
        print(f"\nTesting '{model}'...")
        if test_ollama_connection(model):
            MODEL_NAME = model
            break

    if MODEL_NAME is None:
        print("\n⚠️ No working models found. Please pull a model:")
        print("  ollama pull qwen:0.5b")
        print("  or")
        print("  ollama pull llama3.2:1b")
else:
    print("\n⚠️ No models found. Please install and pull a model:")
    print("  ollama pull qwen:0.5b")

# Configuration
MAX_CONTEXT_LENGTH = 2000  # Smaller for faster testing
TEMPERATURE = 0.0


# In[20]:


# Cell 2: Simple test function to verify model works
def simple_test(model_name: str, question: str, context: str) -> str:
    """Simplest possible test to verify model works."""
    try:
        import ollama

        prompt = f"""Context: {context[:500]}

Question: {question}

Answer with ONE short phrase from the context above:"""

        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 50
            }
        )

        return response['response'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Test with a simple example
if 'MODEL_NAME' in locals() and MODEL_NAME:
    print("\n" + "=" * 60)
    print("SIMPLE MODEL TEST")
    print("=" * 60)

    test_context = "The contract was signed by John Smith and Mary Jones on January 15, 2020."
    test_question = "Who signed the contract?"

    result = simple_test(MODEL_NAME, test_question, test_context)
    print(f"\nTest Question: {test_question}")
    print(f"Test Context: {test_context}")
    print(f"Model Response: '{result}'")

    if "John" in result or "Smith" in result:
        print("\n✓ Model is working correctly!")
    else:
        print("\n⚠️ Model response doesn't match expected output. Check model.")


# In[21]:


# Cell 1: Test different prompt formats to bypass thinking mode
import ollama
import re

MODEL_NAME = "qwen3.5:0.8b"

def test_prompt_format(model: str, prompt: str, description: str) -> str:
    """Test a specific prompt format and return response."""
    print(f"\n--- Testing: {description} ---")
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 200,
                "top_p": 0.9,
            }
        )
        result = response['response'].strip()
        print(f"Response: '{result[:200]}'")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Test context
test_context = "The contract was signed by John Smith and Mary Jones on January 15, 2020."
test_question = "Who signed the contract?"

# Test 1: Direct instruction
prompt1 = f"Answer this question with ONLY the names: {test_question}\n\nText: {test_context}\n\nAnswer:"
test_prompt_format(MODEL_NAME, prompt1, "Direct instruction")

# Test 2: JSON output format
prompt2 = f"""Based on the text, answer the question. Output ONLY as JSON: {{"answer": "..."}}

Text: {test_context}
Question: {test_question}
JSON:"""
test_prompt_format(MODEL_NAME, prompt2, "JSON output")

# Test 3: Single word/phrase instruction
prompt3 = f"""Extract the exact answer from the text. Give ONE short phrase.

Text: {test_context}
Question: {test_question}
Exact answer:"""
test_prompt_format(MODEL_NAME, prompt3, "Exact extraction")

# Test 4: With system prompt (if supported)
prompt4 = f"""<|im_start|>system
You are a direct answer bot. Never explain, never reason, just output the answer as a short phrase.
<|im_end|>
<|im_start|>user
Text: {test_context}
Question: {test_question}
Answer:<|im_end|>
<|im_start|>assistant"""
test_prompt_format(MODEL_NAME, prompt4, "Chat template format")


# ## 6. Main Execution Pipeline

# In[12]:


# Cell 6: Main pipeline - run this last
def main():
    """Main execution function."""
    print("=== Loading Data ===")
    train_df, test_df = load_data()

    print("\n=== Preprocessing ===")
    train_df = prepare_dataframe(train_df, is_train=True)
    test_df = prepare_dataframe(test_df, is_train=False)

    # Optional: Quick validation on training sample
    print("\n=== Quick Validation (optional) ===")
    # Uncomment to run validation (takes time):
    validate_on_train(train_df, sample_size=20)

    # print("\n=== Generating Predictions ===")
    # test_df = batch_predict(test_df)

    # print("\n=== Creating Submission ===")
    # submission = test_df[['id', 'prediction']].copy()
    # submission.columns = ['id', 'answers']  # Match expected format

    # # Save submission
    # output_path = DATA_DIR / "submission.csv"
    # submission.to_csv(output_path, index=False)
    # print(f"Submission saved to: {output_path}")

    # # Show sample predictions
    # print("\n=== Sample Predictions ===")
    # sample_preds = test_df[['question', 'answer_target', 'prediction']].head(10)
    # print(sample_preds.to_markdown(index=False))

    return submission


# In[13]:


# Run the pipeline
if __name__ == "__main__":
    submission_df = main()


# In[ ]:




