import re
from torch.utils.data import Dataset



########################################
# Custom Dataset for GSM8k Data
########################################
class JointGSM8kDataset(Dataset):
    """
    Each sample in the dataframe (data_list) should have two keys:
      - "high_level": full high-level text, e.g.
           "Question: ... High Level Steps: Step 1: ... Step 2: ... Step 3: ... Conclusion:"
      - "low_level": full low-level text, e.g.
           "Question: ... Answer: Step 1: ... Step 2: ... Step 3: ... Conclusion: The answer is ..."

    For each record, we produce two training examples:
      1) High-level example:
           * Target: full high-level text.
           * Input: same as target but with everything after "High Level Steps:" replaced by "<mask>".
      2) Low-level example:
           * Target: full low-level text.
           * Input: same as target but with each step’s details (and the details after "Conclusion:") replaced by "<mask>".
    """
    def __init__(self, data_list, tokenizer, max_length=512, mask_token="<mask>"):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token = mask_token
        # Make sure the mask token is added.
        special_tokens_dict = {"additional_special_tokens": [mask_token]}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def __len__(self):
        return len(self.data_list)

    def process_high_level(self, text):
        """
        Returns (input_text, target_text) for high-level generation.
        Assumes text is like:
           "Question: ... High Level Steps: Step 1: ... Step 2: ... Conclusion:"
        For input, we mask everything after "High Level Steps:".
        """
        target_text = text
        input_text = re.sub(
            r"(High Level Steps:)(.*)",
            r"\1 " + self.mask_token,
            text,
            flags=re.DOTALL
        )
        return input_text, target_text

    def process_low_level(self, text):
        """
        Returns (input_text, target_text) for low-level generation.
        Assumes text is like:
           "Question: ... Answer: Step 1: ... Step 2: ... Conclusion: ..."
        For input, each explanation after a step label (or after "Conclusion:")
        is replaced with "<mask>".
        """
        target_text = text
        # For step regions, preserve the label but mask the explanation.
        input_text = re.sub(
            r"(Step \d+:)(.*?)(?=Step \d+:|Conclusion:|\Z)",
            r"\1 " + self.mask_token,
            text,
            flags=re.DOTALL
        )
        # Also mask everything after "Conclusion:".
        input_text = re.sub(
            r"(Conclusion:)(.*)",
            r"\1 " + self.mask_token,
            input_text,
            flags=re.DOTALL
        )
        return input_text, target_text

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        high_level_full = sample["high_level"]
        low_level_full = sample["low_level"]

        # Process the high-level example.
        hl_input, hl_target = self.process_high_level(high_level_full)
        # Process the low-level example.
        ll_input, ll_target = self.process_low_level(low_level_full)

        # Tokenize each separately.
        enc_hl_inp = self.tokenizer(
            hl_input, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        enc_hl_tgt = self.tokenizer(
            hl_target, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        enc_ll_inp = self.tokenizer(
            ll_input, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        enc_ll_tgt = self.tokenizer(
            ll_target, max_length=self.max_length, truncation=True, return_tensors="pt"
        )

        hl_example = {
            "input_ids": enc_hl_inp["input_ids"].squeeze(0),
            "attention_mask": enc_hl_inp["attention_mask"].squeeze(0),
            "labels": enc_hl_tgt["input_ids"].squeeze(0),
        }
        ll_example = {
            "input_ids": enc_ll_inp["input_ids"].squeeze(0),
            "attention_mask": enc_ll_inp["attention_mask"].squeeze(0),
            "labels": enc_ll_tgt["input_ids"].squeeze(0),
        }
        # Return a list: first high-level, then low-level.
        return [hl_example, ll_example]
