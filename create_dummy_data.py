import pandas as pd
import numpy as np
import os

def create_dummy_dataset(num_examples=100, seq_length=128):
    """Create a small dummy dataset for debugging purposes"""
    
    # Generate random input_ids and attention_masks
    data = []
    for _ in range(num_examples):
        # Generate random sequence with special tokens
        input_ids = np.random.randint(0, 50257, size=seq_length).tolist()
        
        # Insert the special sequence [12, 4971, 5224] somewhere in the middle
        insert_pos = np.random.randint(30, seq_length-10)
        input_ids[insert_pos:insert_pos+3] = [12, 4971, 5224]
        
        # Create attention mask (all 1s for simplicity)
        attention_mask = [1] * seq_length
        
        data.append({
            'input_ids': str(input_ids),  # Store as string to match original format
            'attention_mask': str(attention_mask)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create data_ directory if it doesn't exist
    os.makedirs('data_', exist_ok=True)
    
    # Save to CSV
    output_path = 'data_/dummy_train_set.csv'
    df.to_csv(output_path, index=False)
    print(f"Created dummy dataset with {num_examples} examples at {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_dummy_dataset() 