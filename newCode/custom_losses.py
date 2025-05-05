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

        # Unpack batch
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        clamp_idx = batch['clamp_idx'].cuda()
        
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
                log_score = log_score_fn(xt, sigma)
                
                # Calculate SEDD loss
                raw_loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
                
                # Weight loss by dsigma (SEDD-specific)
                weighted_loss = (dsigma[:, None] * raw_loss)
                
                # Apply attention masking to only train on non-padding tokens
                masked_loss = apply_mask_to_sedd_loss(weighted_loss, attention_mask)
                
            return masked_loss
            
        if train:
            x0 = input_ids
            loss = loss_fn(x0) / accum
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            accum_iter += 1
            total_loss += loss.detach()
            
            if accum_iter == accum:
                accum_iter = 0
                
                state['step'] += 1
                if optimize_fn is not None:
                    optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
            
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                x0 = input_ids
                loss = loss_fn(x0)
                ema.restore(model.parameters())
        
        return loss
        
    return step_fn 