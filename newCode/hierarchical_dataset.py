import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np

class HierarchicalMathDataset(Dataset):
    def __init__(self, csv_path, max_length=1024):
        """
        Args:
            csv_path: Path to the CSV file containing pre-tokenized data
            max_length: Maximum sequence length (default 1024 based on config)
        """
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length
        
        # Convert string representations of lists to actual lists/tensors
        self.data['input_ids'] = self.data['input_ids'].apply(ast.literal_eval)
        self.data['attention_mask'] = self.data['attention_mask'].apply(ast.literal_eval)
        
    def __len__(self):
        return len(self.data)
    
    def find_clamp_index(self, input_ids):
        """Find the index before [12, 4971, 5224] sequence in input_ids"""
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_seq = torch.tensor([12, 4971, 5224], dtype=torch.long)
        
        # Find all occurrences of the first token (12)
        # Add 1 to include token 12 in the clamped sequence
        matches = (input_ids == target_seq[0]).nonzero() + 1
        
        for pos in matches:
            idx = pos.item()
            if idx + len(target_seq) <= len(input_ids):
                if torch.all(input_ids[idx:idx + len(target_seq)] == target_seq):
                    return idx  # Return the index where the sequence starts
                
        # If sequence not found, return None
        return None
    
    def __getitem__(self, idx):
        # Get data from pandas
        item = self.data.iloc[idx]
        
        # Convert to tensors with proper dtypes
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)  # Use long for indices
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.float32)  # Use float for mask
        
        # Find clamp index
        clamp_idx = self.find_clamp_index(input_ids)
        if clamp_idx is None:
            # If sequence not found, set clamp_idx to 0 (no clamping)
            clamp_idx = 0
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'clamp_idx': clamp_idx
        }

def get_math_dataloaders(train_path=None, valid_path=None, batch_size=32, num_workers=4):
    train_loader = None
    if train_path:
        train_dataset = HierarchicalMathDataset(train_path)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    valid_loader = None
    if valid_path:
        valid_dataset = HierarchicalMathDataset(valid_path)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, valid_loader 