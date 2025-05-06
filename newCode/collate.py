import torch
from torch.nn.utils.rnn import pad_sequence

########################################
# Custom Collate Function
########################################
def joint_collate_fn(batch, pad_token_id=0):
    """
    Each sample in batch is a list of two examples [{input_ids, attn_mask, labels}, {input_ids, attn_mask, labels}].
    This function flattens the list of lists into a single list,
    then pads each component (input_ids, attention_mask, labels)
    across all examples in the flattened batch.

    Args:
        batch: A list of samples, where each sample is a list containing two dictionaries.
        pad_token_id: The token ID used for padding.

    Returns:
        A dictionary containing padded tensors for "input_ids", "attention_mask", and "labels".
    """
    flat_examples = [ex for sample in batch for ex in sample] # Flatten: List[Dict]

    input_ids_list = [ex["input_ids"] for ex in flat_examples]
    attention_mask_list = [ex["attention_mask"] for ex in flat_examples]
    labels_list = [ex["labels"] for ex in flat_examples] # labels are the target sequences for SEDD

    # Pad sequences
    # batch_first=True means the output shape will be [batch_size, sequence_length]
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0) # Mask uses 0 for padding
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=pad_token_id) # Pad labels same as input_ids

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels, # This will be used as the 'batch' for SEDD loss
    }

