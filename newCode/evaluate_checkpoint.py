import os
import sys
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import math # Required for math.ceil if used by dataloaders, though not directly in this script's flow

# Add parent directory to path to import from original codebase
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from model import SEDD
import noise_lib
import graph_lib

from hierarchical_dataset import get_math_dataloaders
from custom_losses import get_hierarchical_step_fn

def perform_evaluation(model, dataloader, batch_eval_fn, device):
    """
    Performs a single evaluation pass on the provided model and dataloader.
    """
    model.eval() # Ensure model is in evaluation mode

    total_loss = 0.0
    total_samples = 0
    
    # The batch_eval_fn (from get_hierarchical_step_fn with train=False)
    # expects a 'state' dict with at least a 'model' key.
    eval_run_state = {'model': model} 

    num_samples_in_loader = 0
    try:
        num_samples_in_loader = len(dataloader.dataset)
    except TypeError: # Some dataloaders might not have a dataset with __len__
        print("Could not determine dataset size from dataloader.")


    print(f"\nStarting evaluation on {num_samples_in_loader if num_samples_in_loader > 0 else 'unknown number of'} samples...")
    with torch.no_grad(): # Ensure no gradients are computed
        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        for batch in pbar:
            # Assuming batch items are tensors and get_hierarchical_step_fn handles device transfer
            loss_output = batch_eval_fn(eval_run_state, batch)

            if isinstance(loss_output, torch.Tensor):
                # Assuming 'input_ids' is present and gives batch size
                current_batch_size = batch.get('input_ids', batch.get('image', None)).shape[0] if hasattr(batch.get('input_ids', batch.get('image', None)), 'shape') else 1

                total_loss += loss_output.item() * current_batch_size
                total_samples += current_batch_size
                if total_samples > 0:
                    pbar.set_postfix({'avg_loss': f'{total_loss / total_samples:.4f}', 'batch_loss': f'{loss_output.item():.4f}'})
                else:
                    pbar.set_postfix({'batch_loss': f'{loss_output.item():.4f}'})
            else:
                print(f"Warning: Evaluation step returned non-tensor loss or None: {loss_output}")

    if total_samples == 0:
        print("Warning: No samples processed during evaluation. Returning NaN.")
        return float('nan')
    
    avg_loss = total_loss / total_samples
    print(f"Evaluation finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

@hydra.main(config_path="../configs", config_name="config_inference", version_base="1.1")
def main(cfg: DictConfig):
    print("--- Evaluation Script Started ---")
    print(f"Using configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # --- Paths and Parameters ---
    # Model checkpoint path (expected to be overridden via command line or a specific eval config)
    model_checkpoint_path_str = cfg.get('model_checkpoint_path', None)
    if not model_checkpoint_path_str:
        print("Error: `model_checkpoint_path` not specified in config or command line.")
        sys.exit(1)

    # Validation data path
    valid_data_path_str = cfg.data.get('valid', None)
    if not valid_data_path_str:
        print("Error: `cfg.data.valid` (validation data path) not specified in config.")
        sys.exit(1)
    
    # Make paths absolute if they are relative to project root
    if not os.path.isabs(model_checkpoint_path_str):
        model_checkpoint_path = os.path.join(PROJECT_ROOT, model_checkpoint_path_str)
    else:
        model_checkpoint_path = model_checkpoint_path_str
        
    if not os.path.isabs(valid_data_path_str):
        cfg_data_valid_abs = os.path.join(PROJECT_ROOT, valid_data_path_str)
    else:
        cfg_data_valid_abs = valid_data_path_str

    print(f"Model checkpoint path: {model_checkpoint_path}")
    print(f"Validation data path: {cfg_data_valid_abs}")

    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Checkpoint file not found at {model_checkpoint_path}")
        sys.exit(1)
    if not os.path.exists(cfg_data_valid_abs):
        print(f"Error: Validation data file not found at {cfg_data_valid_abs}")
        sys.exit(1)

    # Batch size for validation
    batch_size = cfg.training.get('batch_size', 32) # Default if not in config
    print(f"Using batch size for validation: {batch_size}")

    # --- Load Tokenizer ---
    print("Loading tokenizer (GPT2TokenizerFast)...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') # Assumes gpt2, adjust if configurable

    # --- Initialize Model ---
    print("Initializing SEDD model...")
    # SEDD(cfg) will use sub-configurations like cfg.model, cfg.sde etc. from the loaded Hydra config.
    score_model = SEDD(cfg).to(device)

    # --- Load Weights ---
    print(f"Loading weights from: {model_checkpoint_path}")
    try:
        loaded_checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        if 'model' in loaded_checkpoint and isinstance(loaded_checkpoint['model'], torch.nn.Module):
            print("Checkpoint detected as full training state. Loading 'model.state_dict()'.")
            score_model.load_state_dict(loaded_checkpoint['model'].state_dict())
        elif isinstance(loaded_checkpoint, dict) and 'state_dict' in loaded_checkpoint and isinstance(loaded_checkpoint['state_dict'], dict):
            print("Checkpoint detected as dictionary with 'state_dict' key. Loading from 'state_dict'.")
            score_model.load_state_dict(loaded_checkpoint['state_dict'])
        elif isinstance(loaded_checkpoint, dict) and not any(isinstance(v, torch.nn.Module) for v in loaded_checkpoint.values()):
            # This handles .pt files that are direct model state_dicts (e.g., saved EMA weights)
            print("Checkpoint detected as a model state_dict. Loading directly.")
            score_model.load_state_dict(loaded_checkpoint)
        else:
            err_msg = "Unrecognized checkpoint format. Could not determine how to load weights. "
            err_msg += f"Loaded object type: {type(loaded_checkpoint)}. "
            if isinstance(loaded_checkpoint, dict):
                err_msg += f"Keys: {list(loaded_checkpoint.keys())}"
            raise ValueError(err_msg)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    score_model.eval() # Ensure model is in evaluation mode after loading weights

    # --- Create Validation Dataloader ---
    print("Creating validation dataloader...")
    # get_math_dataloaders does not require tokenizer to be passed based on train_hierarchical.py
    _, val_loader = get_math_dataloaders(
        train_file=None, # No training data needed
        valid_file=cfg_data_valid_abs,
        batch_size=batch_size
    )
    if val_loader is None:
        print("Failed to create validation dataloader.")
        sys.exit(1)
    print("Validation dataloader created.")

    # --- Initialize Noise and Graph (required for the step function) ---
    print("Initializing noise and graph objects...")
    noise = noise_lib.get_noise(cfg).to(device) # Uses cfg.noise
    graph = graph_lib.get_graph(cfg, device)   # Uses cfg.graph

    # --- Get Validation Batch Evaluation Function ---
    print("Getting validation step function...")
    # train=False, optimize_fn=None, accum=1 are typical for evaluation
    validation_batch_eval_fn = get_hierarchical_step_fn(
        noise=noise,
        graph=graph,
        train=False,
        optimize_fn=None, 
        accum=1 # Accumulation not relevant for eval, set to 1
    )

    # --- Perform Evaluation ---
    avg_val_loss = perform_evaluation(score_model, val_loader, validation_batch_eval_fn, device)

    # --- Print Final Result ---
    print(f"\n--- Final Result ---")
    if not math.isnan(avg_val_loss):
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
    else:
        print("Validation could not be completed or resulted in NaN loss.")
    
    print("--- Evaluation Script Finished ---")

if __name__ == "__main__":
    main() 