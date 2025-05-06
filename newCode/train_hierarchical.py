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

from hierarchical_dataset import get_math_dataloaders
from custom_losses import get_hierarchical_step_fn

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

        # Create output directories (still useful for local logs/plots)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # If W&B is active, maybe incorporate run name/id into output_dir?
        run_id = run.id if run else timestamp
        output_dir = f"outputs/{run_id}"
        checkpoint_dir = os.path.join(output_dir, "checkpoints") # Dir for temporary local saves
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Save configuration locally (optional if using W&B config)
        # We use wandb.config if available, otherwise original cfg
        try:
            if run:
                # wandb.config might not be directly serializable to YAML if complex objects exist
                # Convert back to a simple dict first
                config_dict_to_save = OmegaConf.create(wandb.config.as_dict()) # Convert back to OmegaConf then container
                with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                     OmegaConf.save(config=config_dict_to_save, f=f)
            else:
                 # Assuming original cfg is OmegaConf
                 with open(os.path.join(output_dir, "config.yaml"), "w") as f:
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
        train_loader = get_math_dataloaders(
            cfg_data_train_abs,
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
        # Calculate frequency to save ~100 checkpoints, ensure it's at least 1
        checkpoint_saving_freq = max(1, total_steps // 100) if total_steps > 0 else 100 # Avoid division by zero

        print(f"Dataset size: {num_examples} examples")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training for {num_epochs} epochs, total approx {total_steps} steps")
        print(f"Saving checkpoint artifact every {checkpoint_saving_freq} steps.")

        # Initialize models - PASS ORIGINAL cfg
        score_model = SEDD(cfg).to(device)
        # Read EMA decay specifically
        ema_decay = wandb.config.training['ema'] if run else cfg.training.ema
        ema = ExponentialMovingAverage(
            score_model.parameters(),
            decay=ema_decay
        )

        # Initialize noise and graph - PASS ORIGINAL cfg
        noise = noise_lib.get_noise(cfg).to(device)
        graph = graph_lib.get_graph(cfg, device)

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

        # Training loop
        print("Starting training...")
        global_step = 0

        # For loss tracking
        epoch_losses = []
        # step_log_losses = [] # Can remove if only plotting epoch loss

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

                # --- Checkpoint Saving based on frequency ---
                if global_step > 0 and global_step % checkpoint_saving_freq == 0:
                    print(f"Saving checkpoint artifact at step {global_step}...")

                    # 1. Save Full State
                    full_state_path = os.path.join(checkpoint_dir, f'state_step_{global_step}.pt')
                    try:
                        # torch.save(state, full_state_path) #commented out to avoid excessive memory usage
                        # print(f"Saved local full state to {full_state_path}")
                        if run:
                            artifact_name = f'full_state_step_{global_step}'
                            artifact = wandb.Artifact(
                                name=artifact_name,
                                type='model-state',
                                metadata={'epoch': epoch + 1, 'step': global_step, 'loss': loss.item() if isinstance(loss, torch.Tensor) else None}
                            )
                            artifact.add_file(full_state_path)
                            run.log_artifact(artifact)
                            print(f"Logged full state artifact to W&B: {artifact_name}")
                            # os.remove(full_state_path) # Optional: remove local after upload
                    except Exception as e:
                         print(f"Error saving/logging full state checkpoint at step {global_step}: {e}")

                    # 2. Save EMA Weights
                    ema_weights_path = os.path.join(checkpoint_dir, f'ema_weights_step_{global_step}.pt')
                    try:
                        ema.store(score_model.parameters())
                        ema.copy_to(score_model.parameters())
                        # torch.save(score_model.state_dict(), ema_weights_path) #commented out to avoid excessive memory usage
                        # print(f"Saved local EMA weights to {ema_weights_path}")
                        ema.restore(score_model.parameters()) # IMPORTANT: Restore non-EMA weights

                        if run:
                            artifact_name = f'ema_weights_step_{global_step}'
                            artifact = wandb.Artifact(
                                name=artifact_name,
                                type='model',
                                metadata={'epoch': epoch + 1, 'step': global_step, 'loss': loss.item() if isinstance(loss, torch.Tensor) else None}
                            )
                            artifact.add_file(ema_weights_path)
                            run.log_artifact(artifact)
                            print(f"Logged EMA weights artifact to W&B: {artifact_name}")
                            # os.remove(ema_weights_path) # Optional: remove local after upload
                    except Exception as e:
                        print(f"Error saving/logging EMA weights checkpoint at step {global_step}: {e}")
                        # Ensure restore happens even if logging fails
                        if 'ema' in locals() and 'score_model' in locals():
                             try:
                                 ema.restore(score_model.parameters())
                                 print("Restored non-EMA weights after logging failure.")
                             except Exception as restore_e:
                                 print(f"Error restoring EMA weights after logging failure: {restore_e}")

            # --- End Checkpoint Saving ---

        # End of epoch loop
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        # Handle case where epoch had 0 valid steps/losses
        avg_epoch_loss = epoch_loss / steps_this_epoch if steps_this_epoch > 0 else 0.0
        if steps_this_epoch > 0: # Only append if loss was calculated
             epoch_losses.append(avg_epoch_loss)

        # --- W&B Log Epoch Metrics ---
        if run:
             try:
                 log_data = {
                     'train/epoch_loss': avg_epoch_loss,
                     'epoch': epoch + 1, # Log 1-based epoch index
                     'epoch_duration_sec': epoch_duration,
                 }
                 # Add validation loss logging here if implemented
                 wandb.log(log_data, step=global_step) # Log against global step
             except Exception as wb_log_e:
                 print(f"Warning: W&B epoch logging failed: {wb_log_e}")
        # --- End W&B Log ---

        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s")
        print(f"Average Training Loss: {avg_epoch_loss:.4f}")

        # Save epoch metrics locally
        try:
             with open(os.path.join(log_dir, f"epoch_{epoch+1}_metrics.txt"), "w") as f:
                 f.write(f"Epoch: {epoch+1}\n")
                 f.write(f"Average Training Loss: {avg_epoch_loss:.6f}\n")
                 # Add validation loss here if implemented
                 f.write(f"Duration: {epoch_duration:.2f}s\n")
                 f.write(f"Steps: {steps_this_epoch}\n")
        except Exception as e:
             print(f"Error saving local epoch metrics: {e}")

        # === OLD CHECKPOINT SAVING LOGIC REMOVED ===

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
    if 'epoch_losses' in locals() and epoch_losses: # Check if list exists and is not empty
        for i, loss_val in enumerate(epoch_losses):
            print(f"Epoch {i+1}: {loss_val:.6f}")
    else:
        print("(No complete epochs recorded)")

    # --- Save epoch loss data locally ---
    if 'epoch_losses' in locals() and epoch_losses:
        epoch_loss_path = os.path.join(log_dir, "epoch_losses.json") # log_dir needs to be defined outside the try block or handled
        try:
            # Ensure log_dir exists if an early error occurred before its creation
            if not os.path.exists(log_dir): os.makedirs(log_dir) 
            with open(epoch_loss_path, "w") as f:
                epoch_loss_data = [[i+1, loss_val] for i, loss_val in enumerate(epoch_losses)]
                json.dump(epoch_loss_data, f)
            print(f"Saved epoch loss data to {epoch_loss_path}")
        except Exception as e:
            print(f"Error saving epoch loss data: {e}")
    # --- End saving epoch loss data ---

    # --- Plotting Epoch Loss Curve ---
    if 'epoch_losses' in locals() and epoch_losses:
        plot_path = os.path.join(output_dir, "epoch_loss_curve.png") # output_dir needs definition scope check
        try:
            # Ensure output_dir exists if an early error occurred
            if not os.path.exists(output_dir): os.makedirs(output_dir)
                
            epochs_list = range(1, len(epoch_losses) + 1)
            plt.figure(figsize=(12, 6))
            plt.plot(epochs_list, epoch_losses, marker='o', linestyle='-', label='Average Training Loss per Epoch')
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title("Epoch vs. Average Training Loss")
            if len(epochs_list) <= 20 : # Avoid overcrowding ticks for many epochs
                 plt.xticks(epochs_list)
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_path)
            print(f"Saved epoch loss plot to {plot_path}")

            # --- W&B Log Plot ---
            if run: # Check if run object exists (might have failed init)
                 try:
                     wandb.log({"epoch_loss_curve": wandb.Image(plot_path)}, step=global_step if 'global_step' in locals() else None) # Use final global_step if available
                 except Exception as e:
                     print(f"Error logging plot to W&B: {e}")
            # --- End W&B Log ---
            plt.close()

        except ImportError:
            print("Matplotlib not found. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating/logging plot: {e}")
            if 'plt' in locals(): plt.close()
    else:
         print("No epoch losses recorded, skipping plot generation.")
    # --- End Plotting Epoch Loss Curve ---

    # --- Save final metrics locally ---
    if 'epoch_losses' in locals(): # Check if list exists
        final_metrics_path = os.path.join(output_dir, "final_metrics.txt")
        try:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            with open(final_metrics_path, "w") as f:
                f.write(f"Total Epochs Trained: {len(epoch_losses)}\n") # Use actual number trained
                if epoch_losses:
                    f.write(f"Final Average Loss: {epoch_losses[-1]:.6f}\n")
                f.write("Loss by epoch:\n")
                for i, loss_val in enumerate(epoch_losses):
                    f.write(f"Epoch {i+1}: {loss_val:.6f}\n")
        except Exception as e:
             print(f"Error saving local final metrics: {e}")

    # --- Log final metrics file as W&B artifact (optional) ---
    if run and 'final_metrics_path' in locals() and os.path.exists(final_metrics_path):
        try:
            artifact = wandb.Artifact(name='final_metrics', type='results')
            artifact.add_file(final_metrics_path)
            run.log_artifact(artifact)
            print("Logged final metrics file to W&B.")
        except Exception as e:
            print(f"Error logging final metrics artifact: {e}")

if __name__ == "__main__":
    main() 