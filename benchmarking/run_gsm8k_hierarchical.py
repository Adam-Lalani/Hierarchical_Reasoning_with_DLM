import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Or "true" if you want to risk it, but false is safer for this warning


import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Remove: from load_model import load_model
# Add new imports for local model loading
from omegaconf import OmegaConf
from model.transformer import SEDD # Assuming SEDD is in model/transformer.py
import noise_lib
import graph_lib


import torch

from transformers import GPT2TokenizerFast
from sampling import get_pc_sampler
from datasets import load_dataset
import json
import re
from fractions import Fraction
from tqdm import tqdm
import csv

# --- Configuration for model loading ---
MODEL_CHECKPOINT_PATH = "path/to/your/model_weights.pt"
CONFIG_PATH = "path/to/your/config.yaml"              
# ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load Hydra Config for Model ---
print(f"Loading configuration from: {CONFIG_PATH}")
cfg_abs_path = CONFIG_PATH
if not os.path.isabs(CONFIG_PATH):
    cfg_abs_path = os.path.join(project_root, CONFIG_PATH)

if not os.path.exists(cfg_abs_path):
    print(f"Error: Config file not found at {cfg_abs_path}")
    sys.exit(1)
cfg = OmegaConf.load(cfg_abs_path)

# --- Initialize Model ---
print("Initializing SEDD model...")
model = SEDD(cfg).to(device)

# --- Load Weights ---
print(f"Loading weights from: {MODEL_CHECKPOINT_PATH}")
model_checkpoint_path_abs = MODEL_CHECKPOINT_PATH
if not os.path.isabs(MODEL_CHECKPOINT_PATH):
    model_checkpoint_path_abs = os.path.join(project_root, MODEL_CHECKPOINT_PATH)

if not os.path.exists(model_checkpoint_path_abs):
    print(f"Error: Model checkpoint file not found at {model_checkpoint_path_abs}")
    sys.exit(1)

try:
    loaded_checkpoint = torch.load(model_checkpoint_path_abs, map_location=device)
    if 'model' in loaded_checkpoint and isinstance(loaded_checkpoint['model'], torch.nn.Module):
        print("Checkpoint detected as full training state. Loading 'model.state_dict()'.")
        model.load_state_dict(loaded_checkpoint['model'].state_dict())
    elif isinstance(loaded_checkpoint, dict) and 'state_dict' in loaded_checkpoint and isinstance(loaded_checkpoint['state_dict'], dict):
        print("Checkpoint detected as dictionary with 'state_dict' key. Loading from 'state_dict'.")
        model.load_state_dict(loaded_checkpoint['state_dict'])
    elif isinstance(loaded_checkpoint, dict) and not any(isinstance(v, torch.nn.Module) for v in loaded_checkpoint.values()):
        print("Checkpoint detected as a model state_dict. Loading directly.")
        model.load_state_dict(loaded_checkpoint)
    else:
        err_msg = f"Unrecognized checkpoint format. Loaded object type: {type(loaded_checkpoint)}. "
        if isinstance(loaded_checkpoint, dict):
            err_msg += f"Keys: {list(loaded_checkpoint.keys())}"
        raise ValueError(err_msg)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    sys.exit(1)

model.eval()

# --- Initialize Noise and Graph (required for the sampler) ---
print("Initializing noise and graph objects...")
noise = noise_lib.get_noise(cfg).to(device) # Uses cfg.noise from loaded config
graph = graph_lib.get_graph(cfg, device)   # Uses cfg.graph from loaded config
print("Model, noise, and graph loaded successfully.")


# Load GSM8K test examples
gsm8k = load_dataset("gsm8k", "main", split="test")


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.decode([127]) 

eos_token = tokenizer.eos_token
context_len = 512 # SEDD model's context length
batch_size = 1 # Script is designed for batch_size 1
samples = []

# Helper function to prepare input for the model
def prepare_model_input(prompt_text, tokenizer, context_len, device, batch_size=1):
    tokenized_prompt = tokenizer(prompt_text, truncation=False, padding=False)
    prefix_ids = tokenized_prompt.input_ids

    actual_prefix_len = min(len(prefix_ids), context_len)
    input_locs = list(range(actual_prefix_len))

    if len(prefix_ids) >= context_len:
        padded_input_ids = prefix_ids[:context_len]
    else:
        # Using 0 for padding as implied by original script's [0] * ...
        # Ensure this is consistent with your model's pretraining/fine-tuning
        padding_token_id = 0
        padded_input_ids = prefix_ids + [padding_token_id] * (context_len - len(prefix_ids))
    
    input_tensor = torch.tensor(padded_input_ids, device=device)[None].repeat(batch_size, 1)
    return input_tensor, actual_prefix_len, input_locs

for example in tqdm(gsm8k, desc="Processing examples"):
    question = example["question"].strip()
    answer = example["answer"].strip()  # Used only for evaluation

    # --- Hierarchical Inference ---

    # Pass 1: Generate High-Level Plan
    prompt1 = f"Question: {question} ; High-Level Plan ; "
    input_tensor1, actual_prefix_len1, input_locs1 = prepare_model_input(prompt1, tokenizer, context_len, device, batch_size)

    def proj_fun1(x):
        x[:, input_locs1] = input_tensor1[:, input_locs1]
        return x

    sampler1 = get_pc_sampler(
        graph, noise, (batch_size, context_len), 
        cfg.sampling.predictor, # Use predictor from config
        cfg.sampling.steps,     # Use steps from config
        device=device, proj_fun=proj_fun1
    )

    high_level_plan_text = ""
    with torch.no_grad():
        output1 = proj_fun1(sampler1(model))
        # Decode only the generated part after the prefix
        decoded_output1 = tokenizer.decode(output1[0][actual_prefix_len1:], skip_special_tokens=True)
        high_level_plan_text = decoded_output1.split(eos_token)[0].strip()

    # Pass 2: Generate Low-Level Reasoning (Final Answer)
    prompt2 = f"{prompt1}{high_level_plan_text} ; Low-Level Reasoning ; "
    input_tensor2, actual_prefix_len2, input_locs2 = prepare_model_input(prompt2, tokenizer, context_len, device, batch_size)

    def proj_fun2(x):
        x[:, input_locs2] = input_tensor2[:, input_locs2]
        return x

    sampler2 = get_pc_sampler(
        graph, noise, (batch_size, context_len),
        cfg.sampling.predictor, # Use predictor from config
        cfg.sampling.steps,     # Use steps from config
        device=device, proj_fun=proj_fun2
    )
    
    predicted_answer = ""
    with torch.no_grad():
        output2 = proj_fun2(sampler2(model))
        # Decode only the generated part after the prefix
        decoded_output2 = tokenizer.decode(output2[0][actual_prefix_len2:], skip_special_tokens=True)
        predicted_answer = decoded_output2.split(eos_token)[0].strip()
        
    samples.append({
        "question": question,
        "predicted_answer": predicted_answer, # This is now from the hierarchical process
        "ground_truth": answer,
        "intermediate_high_level_plan": high_level_plan_text, # Optional: for debugging
    })


# Add code to save samples to CSV
if samples: # Ensure samples is not empty
    keys = samples[0].keys()
    with open('gsm8k_samples.csv', 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(samples)
    print("Samples saved to gsm8k_samples.csv")


correct = 0
total = 0

def extract_numeric(text):
    """Extract and normalize the most likely numeric answer from reasoning text."""
    try:
        # Prefer content after '###' if available
        if '###' in text:
            text = text.split('###')[-1]
        
        # Clean up
        text = text.strip()

        # Find all candidates: fractions, decimals, or integers
        candidates = re.findall(r'-?\d+\s*/\s*\d+|-?\d*\.\d+|-?\d+', text)
        if not candidates:
            return None

        # Use last candidate (most likely to be final answer)
        raw = candidates[-1].replace(" ", "")
        value = float(Fraction(raw))  # Normalize everything (e.g., "4/5" => 0.8)
        return round(value, 6)
    except:
        return None


print(samples[0])
print(samples[1])
print(samples[2])
print('\n\n')

for sample in samples:
    pred = extract_numeric(sample['predicted_answer'])
    truth = extract_numeric(sample['ground_truth'])

    if pred is not None and truth is not None and abs(pred - truth) < 1e-4:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0.0
print(f"Robust Numeric Accuracy: {accuracy:.2%}")

print(samples[0])
print(samples[1])
print(samples[2])