import os
import sys
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import GPT2TokenizerFast
import math
import time
from tqdm import tqdm
import yaml
import io # Ensure io is imported
import logging

# 1. Add imports
import wandb
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots on servers
import matplotlib.pyplot as plt
import json # To save the epoch loss data easily

# Add parent directory to path to import from original codebase
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model import SEDD
from model.ema import ExponentialMovingAverage
import utils
import noise_lib
import graph_lib
import losses
from load_model import load_model  # Import the load_model function

from hierarchical_dataset import get_math_dataloaders
from custom_losses import get_hierarchical_step_fn

def evaluate(model_to_eval, dataloader, validation_batch_calculator_fn, epoch_num):
    # The validation_batch_calculator_fn (which is step_fn with train=False from get_hierarchical_step_fn)
    # will internally handle model_to_eval.eval() and also noise.eval() for the noise object it closed over.
    # It will also use train=False for mutils.get_score_fn.
    # It will NOT perform backprop or optimizer steps.

    total_val_loss = 0.0
    num_val_samples = 0
    print(f"\nStarting validation for epoch {epoch_num + 1}...")

    # Create a minimal state dict for the evaluation function if needed by validation_batch_calculator_fn
    # The step_fn from get_hierarchical_step_fn expects a state dict with at least a 'model' key.
    eval_state = {'model': model_to_eval} 

    # model_to_eval.eval() is called inside validation_batch_calculator_fn (when train=False)
    # noise.eval() is also called inside validation_batch_calculator_fn (when train=False)

    with torch.no_grad(): # Ensure no gradients are computed during validation
        val_pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch_num + 1}", leave=False)
        for batch in val_pbar:
            loss_output = validation_batch_calculator_fn(eval_state, batch)

            if isinstance(loss_output, torch.Tensor):
                total_val_loss += loss_output.item() * batch['input_ids'].shape[0] 
                num_val_samples += batch['input_ids'].shape[0]
                val_pbar.set_postfix({'val_loss': f'{loss_output.item():.4f}'})
            else:
                print("Warning: Validation step returned non-tensor loss or None.")
    
    # Set the model passed to evaluate back to train() mode, as the main loop expects it.
    # The noise model that validation_batch_calculator_fn used (from its closure) was set to eval().
    # The main noise object in the training loop will be set to train() before the next training steps.
    model_to_eval.train()
    # Removed: if hasattr(state['noise'], 'train'): state['noise'].train() # state is not in scope

    if num_val_samples == 0:
        print("Warning: No samples processed during validation, returning inf.")
        return float('inf')
    avg_val_loss = total_val_loss / num_val_samples
    print(f"Validation for epoch {epoch_num + 1} finished. Average Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    run = None # Initialize run to None for error handling
    try:
        # --- W&B Initialization ---
        try:
            # Read W&B details from config
            wandb_cfg = cfg.get('wandb', {}) # Get wandb section, default to empty dict
            wandb_entity = wandb_cfg.get('entity')
            wandb_project = wandb_cfg.get('project', 'default-project') # Default project if not set

            if not wandb_entity:
                print("Warning: W&B entity not set in config file (cfg.wandb.entity). Skipping W&B initialization.")
                raise ValueError("W&B entity is required in config.")

            # Convert OmegaConf to dict for W&B logging
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

            # Initialize a W&B run using config values
            run = wandb.init(
                project=wandb_project, 
                entity=wandb_entity, 
                config=config_dict,          # Log your Hydra configuration
                name=f"run-{time.strftime('%Y%m%d-%H%M%S')}", # Optional: custom run name
                job_type="training",         # Optional: categorize run type
                save_code=True,              # Optional: save main script to W&B
            )
            print(f"W&B initialized successfully. Project: {wandb_project}, Entity: {wandb_entity}")

            if run:
                # Update output_dir if W&B run is active
                output_dir = f"outputs/{run.id}"
                log_dir = os.path.join(output_dir, "logs")
                checkpoint_dir = os.path.join(output_dir, "checkpoints")
        except ValueError as ve:
             # Raised if entity is missing
             print(str(ve))
             run = None
        except Exception as e:
            print(f"Error initializing W&B: {e}. Training continues without W&B.")
            run = None
        # --- End W&B Initialization ---


        # Convert relative data paths to absolute paths
        # Handle potential type change after wandb.init
        if run:
             # wandb.config behaves like a dictionary
             train_path = wandb.config.data['train']
             valid_path = wandb.config.data.get('valid', None) # Use .get for optional key
        else:
             # If W&B failed, use original cfg access
             train_path = cfg.data.train
             valid_path = getattr(cfg.data, 'valid', None)

        cfg_data_train_abs = os.path.join(project_root, train_path)
        cfg_data_valid_abs = os.path.join(project_root, valid_path) if valid_path is not None else None

        # Overwrite cfg paths with absolute paths if needed by subsequent code
        # Be careful if cfg object is immutable after wandb.init
        # It might be safer to just use cfg_data_train_abs and cfg_data_valid_abs directly
        # Example: cfg.data.train = cfg_data_train_abs # Might fail if wandb.config is used

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Create local output directories (W&B also creates its own ./wandb/ dir)
        os.makedirs(checkpoint_dir, exist_ok=True) # checkpoint_dir might not be strictly needed if no local saves
        os.makedirs(log_dir, exist_ok=True) # log_dir might not be strictly needed if no local logs

        # Save configuration locally (optional, W&B logs config automatically)
        # You might decide to remove this local config saving too
        try:
            config_to_save_path = os.path.join(output_dir, "config.yaml")
            if run:
                # wandb.config might not be directly serializable to YAML if complex objects exist
                # Convert back to a simple dict first
                config_dict_to_save = OmegaConf.create(wandb.config.as_dict()) # Convert back to OmegaConf then container
                with open(config_to_save_path, "w") as f:
                     OmegaConf.save(config=config_dict_to_save, f=f)
            else:
                 # Assuming original cfg is OmegaConf
                 with open(config_to_save_path, "w") as f:
                     OmegaConf.save(config=cfg, f=f) # Use OmegaConf save
        except ImportError:
             print("PyYAML or OmegaConf missing? Skipping local config saving.")
        except Exception as e:
             print(f"Error saving local config: {e}")

        # Load tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        # Create data loaders using absolute paths
        # Read batch_size specifically
        batch_size_val = wandb.config.training['batch_size'] if run else cfg.training.batch_size
        train_loader, val_loader = get_math_dataloaders(
            cfg_data_train_abs,
            cfg_data_valid_abs,
            batch_size=batch_size_val 
        )
        # Add validation loader here if you implement it

        # Calculate total steps and checkpoint frequency
        num_examples = len(train_loader.dataset)
        # Use batch_size_val already determined
        steps_per_epoch = math.ceil(num_examples / batch_size_val)
        # Read num_epochs from config, default to 10000
        # Use get with default, access appropriate config object
        num_epochs = wandb.config.training.get('num_epochs', 10000) if run else cfg.training.get('num_epochs', 10000)

        total_steps = steps_per_epoch * num_epochs
        
        # --- Calculate checkpoint frequency for approximately 10 saves across the run --- 
        if total_steps > 0:
            checkpoint_saving_freq = max(1, total_steps // 10) # Ensure frequency is at least 1
        else:
            checkpoint_saving_freq = 1 # Default to saving every step if total_steps is 0 or unknown
            logging.warning("Total steps calculated as 0. Check epoch/batch size config. Defaulting checkpoint freq to 1.")
        # --- End frequency calculation ---

        print(f"Dataset size: {num_examples} examples")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training for {num_epochs} epochs, total approx {total_steps} steps")
        print(f"Saving checkpoint artifact every {checkpoint_saving_freq} steps.")

        # Initialize models - PASS ORIGINAL cfg
        # Replace model initialization with loading pretrained model
        model_path = "louaaron/sedd-medium"
        print(f"Loading pretrained model from: {model_path}")
        score_model, graph, noise = load_model(model_path, device)
        score_model.config.length = 512
        score_model.to(device)
        
        # Read EMA decay specifically
        ema_decay = wandb.config.training['ema'] if run else cfg.training.ema
        ema = ExponentialMovingAverage(
            score_model.parameters(),
            decay=ema_decay
        )

        # Noise and graph are already initialized by load_model
        # No need for these lines:
        # noise = noise_lib.get_noise(cfg).to(device)
        # graph = graph_lib.get_graph(cfg, device)

        # Setup optimization - PASS ORIGINAL cfg
        optimizer = losses.get_optimizer(cfg, score_model.parameters())
        scaler = torch.cuda.amp.GradScaler()

        # Get optimization function from the original codebase - PASS ORIGINAL cfg
        optimize_fn = losses.optimization_manager(cfg)

        # Create state dict (ensure model is correctly referenced)
        state = {
            'optimizer': optimizer,
            'model': score_model, # Reference to the model object
            'ema': ema,
            'noise': noise,
            'step': 0,
            'scaler': scaler,
            'epoch': 0
        }

        # Get training step function
        # Read accum specifically
        accum_val = wandb.config.training['accum'] if run else cfg.training.accum
        train_step_fn = get_hierarchical_step_fn(
            noise,
            graph,
            train=True,
            optimize_fn=optimize_fn,
            accum=accum_val
        )

        # Get the validation batch evaluation function (only if val_loader is successfully created)
        validation_batch_eval_fn = None # Initialize to None
        if valid_path: # Check if valid_path was successfully created before defining this
            validation_batch_eval_fn = get_hierarchical_step_fn(
                noise, graph, train=False, optimize_fn=None, accum=1 
            )

        # Training loop
        print("Starting training...")
        global_step = 0

        # For loss tracking
        epoch_train_losses = []
        epoch_val_losses = []

        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)

        for epoch in epoch_pbar:
            state['epoch'] = epoch
            epoch_start_time = time.time()
            epoch_loss = 0.0
            steps_this_epoch = 0

            # Reset iterator for each epoch
            train_iter = iter(train_loader)

            # Create progress bar for steps within epoch
            step_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)

            # Training phase for the epoch
            score_model.train() # Ensure model is in training mode for the training loop part
            if hasattr(noise, 'train'): noise.train() # Ensure noise is in training mode

            for step_in_epoch in step_pbar:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    print("Warning: Ran out of data before completing epoch cycle.")
                    break

                loss = train_step_fn(state, batch) # This should update state['step']
                global_step = state['step']

                # Accumulate loss for epoch average
                if isinstance(loss, torch.Tensor):
                    epoch_loss += loss.item()
                else:
                    # Handle cases where train_step_fn might not return loss (e.g. accumulation steps)
                    pass # Or log differently if needed
                steps_this_epoch += 1

                # Update step progress bar
                step_pbar.set_postfix({'loss': f'{loss.item():.4f}' if isinstance(loss, torch.Tensor) else 'N/A'})

                # Logging
                # Read log_freq specifically
                log_freq = wandb.config.training['log_freq'] if run else cfg.training.log_freq
                if global_step % log_freq == 0 and isinstance(loss, torch.Tensor):
                    avg_loss_so_far = epoch_loss / steps_this_epoch if steps_this_epoch > 0 else loss.item()
                    epoch_pbar.set_postfix({'avg_loss': f'{avg_loss_so_far:.4f}'})

                    # --- W&B Log Step Loss ---
                    if run:
                         try: # Add try-except for W&B logging robustness
                             wandb.log({'train/step_loss': loss.item()}, step=global_step) # Log against global step
                         except Exception as wb_log_e:
                             print(f"Warning: W&B step logging failed: {wb_log_e}")
                    # --- End W&B Log ---

                # --- Checkpoint Saving based on frequency (to W&B buffer) ---
                if global_step > 0 and global_step % checkpoint_saving_freq == 0:
                    print(f"\nLogging checkpoint artifact directly to W&B at step {global_step} (no local disk save for model weights)...")

                    # 1. Save Full State, Log Artifact, then Remove Local File
                    full_state_path = os.path.join(checkpoint_dir, f'state_step_{global_step}.pt')
                    try:
                        if run:
                            # Save locally first
                            torch.save(state, full_state_path)
                            # print(f"Temporarily saved full state to {full_state_path}") # Optional debug print

                            artifact_name = f'full_state_step_{global_step}'
                            artifact_state = wandb.Artifact(
                                name=artifact_name,
                                type='model-state',
                                metadata={'epoch': epoch + 1, 'step': global_step, 'loss': loss.item() if isinstance(loss, torch.Tensor) else None}
                            )
                            artifact_state.add_file(full_state_path) # Add the local file path
                            run.log_artifact(artifact_state)
                            print(f"Logged full state artifact to W&B: {artifact_name}")

                            # Remove local file after successful upload
                            os.remove(full_state_path)
                            # print(f"Removed local file: {full_state_path}") # Optional debug print
                        else:
                             print("Skipping full state artifact logging: W&B run not active.")
                    except Exception as e:
                         print(f"Error saving/logging full state artifact at step {global_step}: {e}")
                         # Clean up local file if save succeeded but logging failed, and file exists
                         if os.path.exists(full_state_path):
                             try: os.remove(full_state_path) 
                             except Exception as e_rem: print(f"Error removing temp file {full_state_path}: {e_rem}")

                    # 2. Save EMA Weights, Log Artifact, then Remove Local File
                    ema_weights_path = os.path.join(checkpoint_dir, f'ema_weights_step_{global_step}.pt')
                    try:
                        if run:
                            ema.store(score_model.parameters())
                            ema.copy_to(score_model.parameters())
                            
                            # Save locally first
                            torch.save(score_model.state_dict(), ema_weights_path)
                            # print(f"Temporarily saved EMA weights to {ema_weights_path}") # Optional debug print
                            
                            artifact_name_ema = f'ema_weights_step_{global_step}'
                            artifact_ema = wandb.Artifact(
                                name=artifact_name_ema,
                                type='model',
                                metadata={'epoch': epoch + 1, 'step': global_step, 'loss': loss.item() if isinstance(loss, torch.Tensor) else None}
                            )
                            artifact_ema.add_file(ema_weights_path) # Add the local file path
                            run.log_artifact(artifact_ema)
                            print(f"Logged EMA weights artifact to W&B: {artifact_name_ema}")

                            # Remove local file after successful upload
                            os.remove(ema_weights_path)
                            # print(f"Removed local file: {ema_weights_path}") # Optional debug print

                            ema.restore(score_model.parameters()) # IMPORTANT: Restore non-EMA weights
                        else:
                             print("Skipping EMA weights artifact logging: W&B run not active.")
                    except Exception as e:
                        print(f"Error saving/logging EMA weights artifact at step {global_step}: {e}")
                        # Clean up local file if save succeeded but logging failed, and file exists
                        if os.path.exists(ema_weights_path):
                            try: os.remove(ema_weights_path)
                            except Exception as e_rem: print(f"Error removing temp file {ema_weights_path}: {e_rem}")
                        # Ensure restore happens even if logging fails
                        if 'ema' in locals() and 'score_model' in locals():
                             try:
                                 ema.restore(score_model.parameters())
                                 print("Restored non-EMA weights after artifact logging failure.")
                             except Exception as restore_e:
                                 print(f"Error restoring EMA weights: {restore_e}")
                # --- End Checkpoint Saving ---

            # End of step_in_epoch loop

            avg_epoch_loss = epoch_loss / steps_this_epoch if steps_this_epoch > 0 else 0.0
            epoch_train_losses.append(avg_epoch_loss)

            current_val_loss = None # Initialize
            print(f"--- Epoch {epoch+1}: Checking for validation --- ") # Added Log
            # Ensure both val_loader and validation_batch_eval_fn exist before evaluating
            # Note: The original attached code might have had older condition `if valid_path...`, ensure it uses `val_loader` now.
            if 'val_loader' in locals() and val_loader and validation_batch_eval_fn:
                print(f"Epoch {epoch+1}: Starting evaluation step.") # Added Log
                # Note: `evaluate` function will handle model.eval() and model.train() internally
                current_val_loss = evaluate(score_model, val_loader, validation_batch_eval_fn, epoch)
                print(f"Epoch {epoch+1}: Evaluation function returned: {current_val_loss}") # Added Log
                if current_val_loss is not None and current_val_loss != float('inf'): 
                    print(f"Epoch {epoch+1}: Appending valid validation loss {current_val_loss:.4f} to local list.") # Added Log
                    epoch_val_losses.append(current_val_loss)
                else:
                    print(f"Epoch {epoch+1}: Evaluation returned invalid loss: {current_val_loss}") # Added Log
            elif 'val_loader' in locals() and val_loader and not validation_batch_eval_fn:
                 print(f"Epoch {epoch+1}: val_loader exists but validation_batch_eval_fn is missing. Skipping evaluation.") # Added Log
            else: # val_loader is None or doesn't exist
                 print(f"Epoch {epoch+1}: val_loader not available. Skipping evaluation.") # Added Log

            # End of epoch loop 
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            # Handle case where epoch had 0 valid steps/losses
            avg_epoch_loss = epoch_loss / steps_this_epoch if steps_this_epoch > 0 else 0.0
            if steps_this_epoch > 0: # Only append if loss was calculated
                epoch_train_losses.append(avg_epoch_loss)

            # --- W&B Log Epoch Metrics ---
            if run:
                print(f"--- Epoch {epoch+1}: Preparing to log metrics to W&B --- ") # Added Log
                try:
                    log_data = {
                        'train/epoch_loss': avg_epoch_loss,
                        'epoch': epoch + 1, # Log 1-based epoch index
                        'epoch_duration_sec': epoch_duration,
                    }
                    # Add validation loss to the log_data dictionary if it was computed
                    if current_val_loss is not None and current_val_loss != float('inf'):
                        print(f"Epoch {epoch+1}: Adding 'valid/epoch_loss': {current_val_loss:.4f} to W&B log dictionary.") # Added Log
                        log_data['valid/epoch_loss'] = current_val_loss
                    else:
                        print(f"Epoch {epoch+1}: No valid validation loss ({current_val_loss}) to add to W&B log dictionary.") # Added Log
                    
                    print(f"Epoch {epoch+1}: Calling wandb.log() with data: {log_data}") # Added Log
                    wandb.log(log_data, step=global_step) # Log against global step
                    print(f"Epoch {epoch+1}: wandb.log() call executed.") # Added Log
                except Exception as wb_log_e:
                    # Original warning kept, but added print indicates the try block was entered.
                    print(f"Warning: W&B epoch logging failed within try block: {wb_log_e}")
            # --- End W&B Log ---

            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s")
            print(f"Average Training Loss: {avg_epoch_loss:.4f}")
            # Add a print for validation loss if it was computed
            if current_val_loss is not None and current_val_loss != float('inf'):
                print(f"Average Validation Loss: {current_val_loss:.4f}")
            elif cfg_data_valid_abs: # Only print this if validation was expected
                print("Validation not performed or resulted in invalid loss this epoch.")

    finally:
        # --- Finish W&B Run ---
        if run:
            print("Finishing W&B run...")
            try:
                 run.finish()
                 print("W&B run finished.")
            except Exception as e:
                 print(f"Error finishing W&B run: {e}")
        # --- End Finish ---

    print("Training loop finished execution (may have ended due to error or completion).")
    print("Final Loss by epoch recorded (average):") # These results are from epochs that completed *before* any potential error
    if 'epoch_train_losses' in locals() and epoch_train_losses: # Check if list exists and is not empty
        for i, loss_val in enumerate(epoch_train_losses):
            print(f"Epoch {i+1}: {loss_val:.6f}")
    else:
        print("(No complete epochs recorded)")

if __name__ == "__main__":
    main() 