import torch
import torch.nn.functional as F
from train_utils import create_clamp_mask, apply_clamping, apply_mask_to_sedd_loss
from model import utils as mutils

def get_hierarchical_step_fn(noise, graph, train, optimize_fn=None, accum=1):
    """
    Modified step function for hierarchical math training with SEDD loss
    Args:
        noise: Noise model
        graph: Graph model
        train: Whether in training mode
        optimize_fn: Optimization function
        accum: Gradient accumulation steps
    Returns:
        Step function that handles clamping and masked loss
    """
    accum_iter = 0
    total_loss = 0
    
    def step_fn(state, batch):
        nonlocal accum_iter
        nonlocal total_loss
        
        model = state['model']
        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            model.train()
            noise.train()
        else:
            model.eval()
            noise.eval()

        # Unpack batch and ensure proper types
        input_ids = batch['input_ids'].to(torch.long).cuda()  # Ensure long type for indices
        attention_mask = batch['attention_mask'].to(torch.float32).cuda()
        clamp_idx = batch['clamp_idx']
        if isinstance(clamp_idx, torch.Tensor):
            clamp_idx = clamp_idx.to(torch.long).cuda()
        else:
            clamp_idx = torch.tensor(clamp_idx, dtype=torch.long).cuda()
        
        # Create clamping mask
        clamp_mask = create_clamp_mask(clamp_idx, input_ids.shape, input_ids.device)
        
        def loss_fn(x0):
            # Sample noise level
            t = (1 - 1e-3) * torch.rand(x0.shape[0], device=x0.device) + 1e-3
            sigma, dsigma = noise(t)
            
            with torch.cuda.amp.autocast():
                # Sample transition (perturb the tokens)
                xt = graph.sample_transition(x0, sigma[:, None])
                
                # Apply clamping to noised tokens - preserve the question part
                xt = apply_clamping(xt, x0, clamp_mask)
                
                # Get model prediction
                log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
                log_score = log_score_fn(xt.to(torch.long), sigma)  # Ensure long type for indices
                
                # Calculate SEDD loss
                raw_loss = graph.score_entropy(log_score, sigma[:, None], xt.to(torch.long), x0.to(torch.long))  # Ensure long type
                
                # Weight loss by dsigma (SEDD-specific)
                weighted_loss = (dsigma[:, None] * raw_loss)
                
                # Apply attention masking to only train on non-padding tokens
                masked_loss = apply_mask_to_sedd_loss(weighted_loss, attention_mask)
                
            return masked_loss
        
        # Forward pass and loss calculation
        loss = loss_fn(input_ids) / accum
        
        # Backward pass and optimization
        if train:
            scaler.scale(loss).backward()
            accum_iter += 1
            
            if accum_iter == accum:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_iter = 0
                state['step'] += 1
        
        return loss
    
    return step_fn 