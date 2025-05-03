import pandas as pd
from transformers import GPT2TokenizerFast
import os
from tqdm import tqdm

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv_path = os.path.join(script_dir, '../data/train_data_raw.csv')
output_dir = os.path.join(script_dir, '../data')
uniform_len_csv_path = os.path.join(output_dir, 'train_uniform_len.csv')
len_prepend_csv_path = os.path.join(output_dir, 'train_len_prepend.csv')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_str = tokenizer.pad_token
# print(f"Using padding token: '{pad_token_str}'")


# Load the raw training data
try:
    df = pd.read_csv(input_csv_path)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_csv_path}")
    exit()

# Drop rows with missing values in 'high_level' or 'low_level'
df.dropna(subset=['high_level', 'low_level'], inplace=True)

print("Tokenizing data (this might take a while)...")
# Tokenize and get lengths for all rows
tqdm.pandas(desc="Tokenizing High Level")
df['hl_tokens'] = df['high_level'].progress_apply(lambda x: tokenizer.encode(str(x)))
tqdm.pandas(desc="Tokenizing Low Level")
df['ll_tokens'] = df['low_level'].progress_apply(lambda x: tokenizer.encode(str(x)))

df['hl_len'] = df['hl_tokens'].apply(len)
df['ll_len'] = df['ll_tokens'].apply(len)

# --- Generate train_uniform_len.csv ---
print("Processing for train_uniform_len.csv...")

# Find max lengths
max_len_hl = df['hl_len'].max()
max_len_ll = df['ll_len'].max()
print(f"Max high_level token length: {max_len_hl}")
print(f"Max low_level token length: {max_len_ll}")

# Pad sequences
padded_hl = []
padded_ll = []

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Padding Sequences"):
    hl_text = str(row['high_level'])
    ll_text = str(row['low_level'])
    hl_len = row['hl_len']
    ll_len = row['ll_len']

    pad_count_hl = max_len_hl - hl_len
    pad_count_ll = max_len_ll - ll_len

    padded_hl_text = hl_text + (pad_token_str * pad_count_hl)
    padded_ll_text = ll_text + (pad_token_str * pad_count_ll)

    padded_hl.append(padded_hl_text)
    padded_ll.append(padded_ll_text)

# Create uniform length DataFrame
df_uniform = pd.DataFrame({
    'high_level_padded': padded_hl,
    'low_level_padded': padded_ll
})

print(f"Saving uniform length data to {uniform_len_csv_path}...")
df_uniform.to_csv(uniform_len_csv_path, index=False)
print("Saved train_uniform_len.csv")


# --- Generate train_len_prepend.csv ---
print("Processing for train_len_prepend.csv...")

prepended_hl = []
prepended_ll = []

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Prepending Lengths"):
    hl_text = str(row['high_level'])
    ll_text = str(row['low_level'])
    hl_len = row['hl_len']
    ll_len = row['ll_len']

    prepended_hl_text = f"[{hl_len}] {hl_text}"
    prepended_ll_text = f"[{ll_len}] {ll_text}"

    prepended_hl.append(prepended_hl_text)
    prepended_ll.append(prepended_ll_text)

# Create length prepend DataFrame
df_prepend = pd.DataFrame({
    'high_level_prepended': prepended_hl,
    'low_level_prepended': prepended_ll
})

print(f"Saving length prepend data to {len_prepend_csv_path}...")
df_prepend.to_csv(len_prepend_csv_path, index=False)
print("Saved train_len_prepend.csv")

print("Script finished.")
