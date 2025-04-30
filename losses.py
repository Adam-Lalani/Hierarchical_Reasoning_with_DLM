import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, 
                  final_answer_weight=1.0, 
                  final_answer_token_id=None, 
                  pad_token_id=0):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        'batch' here refers to the target sequence (labels from the dataset).
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        
        # Calculate base score entropy loss per position
        # Shape: [B, L]
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        # --- Loss Weighting --- 
        weights = torch.ones_like(loss)
        if final_answer_token_id is not None and final_answer_weight != 1.0:
            # Find where the final answer token occurs in the target sequence
            # We assume the relevant part starts AFTER the final_answer_token_id
            # Create a mask for tokens that are part of the final answer
            answer_mask = torch.zeros_like(batch, dtype=torch.bool)
            # Find the first occurrence of the token_id for each sequence in the batch
            match_indices = (batch == final_answer_token_id).int().argmax(dim=1)
            # Check if the token actually exists in each sequence
            found_mask = (batch == final_answer_token_id).any(dim=1)
            
            # For sequences where the token was found, create a range mask starting after it
            rows_found = torch.where(found_mask)[0]
            cols_start = match_indices[found_mask] + 1 # Start weighting *after* the marker token
            seq_len = batch.shape[1]
            for i, row in enumerate(rows_found):
                # Only apply weight from the token onwards, excluding padding
                non_pad_mask = (batch[row, cols_start[i]:] != pad_token_id)
                answer_mask[row, cols_start[i]:][non_pad_mask] = True

            # Apply the weight
            weights[answer_mask] = final_answer_weight

        # Apply weights to the loss (element-wise)
        weighted_loss = loss * weights
        # --- End Loss Weighting ---

        # Weight by noise derivative and sum over sequence length
        final_loss = (dsigma[:, None] * weighted_loss).sum(dim=-1)

        return final_loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, 
                final_answer_weight=1.0, 
                final_answer_token_id=None, 
                pad_token_id=0):
    # Pass weighting params to loss_fn
    loss_fn = get_loss_fn(noise, graph, train, 
                          final_answer_weight=final_answer_weight,
                          final_answer_token_id=final_answer_token_id,
                          pad_token_id=pad_token_id)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            # 'batch' here is the target sequence (labels)
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else: # Evaluation mode
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                # Note: Using the same loss_fn for evaluation, which includes weighting
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn