import pandas as pd
import ast
import sys
import os

def clean_and_validate_csv(input_path, output_path):
    """
    Clean and validate the CSV file by:
    1. Properly handling quotes and escapes
    2. Ensuring all rows have valid JSON-like strings for input_ids and attention_mask
    3. Skipping or fixing malformed rows
    """
    print(f"Reading file: {input_path}")
    
    # Read the file in chunks to handle large files
    chunk_size = 1000
    chunks = []
    
    try:
        # Use Python engine which is more forgiving with malformed CSV
        for chunk in pd.read_csv(input_path, chunksize=chunk_size, on_bad_lines='warn', 
                               quoting=1, engine='python'):
            # Validate each row's input_ids and attention_mask
            valid_rows = []
            for idx, row in chunk.iterrows():
                try:
                    # Try to parse the lists
                    input_ids = ast.literal_eval(row['input_ids'])
                    attention_mask = ast.literal_eval(row['attention_mask'])
                    
                    # Ensure they are valid lists
                    if isinstance(input_ids, list) and isinstance(attention_mask, list):
                        valid_rows.append(row)
                except Exception as e:
                    print(f"Skipping invalid row at index {idx}: {str(e)}")
                    continue
            
            if valid_rows:
                chunks.append(pd.DataFrame(valid_rows))
    
        # Combine all valid chunks
        if chunks:
            cleaned_df = pd.concat(chunks, ignore_index=True)
            print(f"Writing cleaned data to: {output_path}")
            cleaned_df.to_csv(output_path, index=False, quoting=1)
            print(f"Successfully cleaned data. Original rows: {sum(1 for _ in open(input_path))-1}, Clean rows: {len(cleaned_df)}")
        else:
            print("No valid data found in the input file")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Create data_ directory if it doesn't exist
    if not os.path.exists('data_'):
        os.makedirs('data_')
    
    # Clean both train and validation sets
    train_input = "data_/train_set.csv"
    train_output = "data_/train_set_clean.csv"
    
    if os.path.exists(train_input):
        success = clean_and_validate_csv(train_input, train_output)
        if success:
            # Backup original and replace with cleaned version
            os.rename(train_input, train_input + ".bak")
            os.rename(train_output, train_input)
            print("Successfully replaced original file with cleaned version")
    else:
        print(f"Training file not found: {train_input}") 