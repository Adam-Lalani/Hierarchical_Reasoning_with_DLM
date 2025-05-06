import torch
import torch.nn.functional as F

def create_clamp_mask(clamp_idx, shape, device):
    """
    Create a mask for clamping based on clamp indices
    Args:
        clamp_idx: Tensor of indices [batch_size]
        shape: Tuple of (batch_size, seq_len)
        device: Target device
    Returns:
        Binary mask where 1 indicates positions after clamp_idx
    """
    batch_size, seq_len = shape
    pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    clamp_idx = clamp_idx.unsqueeze(1).expand(-1, seq_len)
    return (pos >= clamp_idx).float()

def apply_clamping(x, x0, clamp_mask):
    """
    Apply clamping to the input tensor
    Args:
        x: Tensor to clamp [batch_size, seq_len, ...]
        x0: Original input tensor [batch_size, seq_len, ...]
        clamp_mask: Binary mask [batch_size, seq_len]
    Returns:
        Tensor with values from x0 where clamp_mask is 0, and x where clamp_mask is 1
    """
    clamp_mask = clamp_mask.unsqueeze(-1) if x.dim() > clamp_mask.dim() else clamp_mask
    return x * clamp_mask + x0 * (1 - clamp_mask)

def masked_loss(logits, targets, attention_mask):
    """
    Calculate loss only on positions where attention_mask is 1
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        targets: Target indices [batch_size, seq_len]
        attention_mask: Binary mask [batch_size, seq_len]
    Returns:
        Average loss over masked positions
    """
    # Calculate cross entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    ).view_as(targets)
    
    # Apply attention mask
    masked_loss = loss * attention_mask
    
    # Average over non-masked positions
    return masked_loss.sum() / attention_mask.sum()

def apply_mask_to_sedd_loss(loss, attention_mask):
    """
    Apply attention mask to SEDD's score_entropy loss
    
    Args:
        loss: Loss tensor from graph.score_entropy [batch_size, seq_len]
        attention_mask: Binary mask [batch_size, seq_len]
    
    Returns:
        Masked loss tensor averaged over non-masked positions
    """
    # Apply attention mask
    masked_loss = loss * attention_mask
    
    # Sum over sequence dimension, then average over batch size
    # using the sum of attention mask weights
    return masked_loss.sum() / attention_mask.sum() 