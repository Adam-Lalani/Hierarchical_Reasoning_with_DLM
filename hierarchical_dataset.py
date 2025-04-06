import re
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
import torch

class HierarchicalSEDDDataset(Dataset):
    def __init__(self, data, stage="high_level", seq_len=1024):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.data = data
        self.seq_len = seq_len
        self.stage = stage
        self.mask_token_id = self.tokenizer.pad_token_id or 0

    def __len__(self):
        return len(self.data)

    def _find_token_range(self, subtext, fulltext):
        """Return the start/end token positions of `subtext` inside `fulltext`."""
        sub_ids = self.tokenizer(subtext, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(fulltext, add_special_tokens=False).input_ids

        for i in range(len(full_ids) - len(sub_ids) + 1):
            if full_ids[i:i+len(sub_ids)] == sub_ids:
                return list(range(i, i+len(sub_ids)))
        return []

    def __getitem__(self, idx):
        text = self.data[idx]
        input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        input_ids = input_ids[:self.seq_len]
        pad_len = self.seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len

        # Determine clamped token indices
        clamp_locs = []

        if self.stage == "high_level":
            clamp_key = "High Level Steps:"
            clamp_range = self._find_token_range(text[:text.find(clamp_key) + len(clamp_key)], text)
            clamp_locs += clamp_range

        elif self.stage == "low_level":
            # Clamp everything up through the "Answer:" header
            answer_index = text.find("Answer:")
            if answer_index == -1:
                raise ValueError("Missing 'Answer:' in input")
            pre_answer = text[:answer_index + len("Answer:")]
            clamp_locs += self._find_token_range(pre_answer, text)

            # Also clamp all "Step X: ... ;" headers
            step_headers = re.findall(r"Step \d+:.*?;", text)
            for header in step_headers:
                clamp_locs += self._find_token_range(header, text)
        else:
            raise ValueError("Invalid stage")

        clamp_locs = sorted(set(clamp_locs))  # remove duplicates
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "clamp_locs": torch.tensor(clamp_locs, dtype=torch.long)
        }