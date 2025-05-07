import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Or "true" if you want to risk it, but false is safer for this warning


import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from load_model import load_model


import torch

from transformers import GPT2TokenizerFast
from sampling import get_pc_sampler
from datasets import load_dataset
import json
import re
from fractions import Fraction
from tqdm import tqdm
import csv

# 
device = torch.device('cuda')

model_path = "louaaron/sedd-small"
model, graph, noise = load_model(model_path, device)


# Load GSM8K test examples
gsm8k = load_dataset("gsm8k", "main", split="test")


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

eos_token = tokenizer.eos_token 
context_len = 512
batch_size = 1
samples = []

for example in tqdm(gsm8k, desc="Processing examples"):
    
    question = example["question"].strip()
    answer = example["answer"].strip()  # Used only for evaluation
   

    # Create full sequence, pad rest
    prefix_ids = tokenizer(question).input_ids
    input_ids = prefix_ids + [0] * (context_len - len(prefix_ids))
    input_locs = list(range(len(prefix_ids)))  # Clamp question tokens

    input_tensor = torch.tensor(input_ids, device="cuda")[None].repeat(batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_tensor[:, input_locs]
        return x

    sampler = get_pc_sampler(
        graph, noise, (batch_size, context_len), 'analytic', 512,
        device=device, proj_fun=proj_fun
    )

    with torch.no_grad():
        output = proj_fun(sampler(model))
        decoded = tokenizer.decode(output[0][len(prefix_ids):], skip_special_tokens=True)
        predicted_answer = decoded.split(eos_token)[0].strip()
        samples.append({
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": answer
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