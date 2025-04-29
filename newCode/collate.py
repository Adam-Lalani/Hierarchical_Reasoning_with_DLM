import torch

########################################
# Custom Collate Function
########################################
def joint_collate_fn(batch):
    """
    Each sample in batch is a list of two examples.
    This function flattens the list of lists into a single list
    so the final batch contains 2*N examples.
    """

    flat_examples = [ex for sample in batch for ex in sample]
    input_ids_list = [ex["input_ids"] for ex in flat_examples]
    attention_mask_list = [ex["attention_mask"] for ex in flat_examples]
    labels_list = [ex["labels"] for ex in flat_examples]

   
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

