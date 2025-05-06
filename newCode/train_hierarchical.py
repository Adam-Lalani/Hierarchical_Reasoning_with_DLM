import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast
import math
import time
from tqdm import tqdm
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
def main(cfg):
    # Convert relative data paths to absolute paths
    cfg.data.train = os.path.join(project_root, cfg.data.train)
    if cfg.data.valid is not None:  # Only join paths if valid path exists
        cfg.data.valid = os.path.join(project_root, cfg.data.valid)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directories
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"outputs/{timestamp}"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Create data loaders
    train_loader = get_math_dataloaders(
        cfg.data.train,
        batch_size=cfg.training.batch_size
    )
    
    # Calculate total number of steps for 2 epochs
    num_examples = len(train_loader.dataset)
    steps_per_epoch = math.ceil(num_examples / cfg.training.batch_size)
    num_epochs = 10000  # Training epochs
    total_steps = steps_per_epoch * num_epochs
    print(f"Dataset size: {num_examples} examples")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Training for {num_epochs} epochs, total {total_steps} steps")
    
    # Initialize models
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(
        score_model.parameters(),
        decay=cfg.training.ema
    )
    
    # Initialize noise and graph
    noise = noise_lib.get_noise(cfg).to(device)
    graph = graph_lib.get_graph(cfg, device)
    
    # Setup optimization
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    scaler = torch.cuda.amp.GradScaler()

    # Get optimization function from the original codebase
    optimize_fn = losses.optimization_manager(cfg)
    
    # Create state dict
    state = {
        'optimizer': optimizer,
        'model': score_model,
        'ema': ema,
        'noise': noise,
        'step': 0,
        'scaler': scaler,
        'epoch': 0
    }
    
    # Get training step function with the original optimization function
    train_step_fn = get_hierarchical_step_fn(
        noise,
        graph,
        train=True,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    
    # For loss tracking
    epoch_losses = []
    
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
        step_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}", position=1, leave=False)
        
        for step in step_pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                # This shouldn't happen within an epoch, but just in case
                print("Warning: Ran out of data before completing an epoch")
                break
                
            loss = train_step_fn(state, batch)
            global_step = state['step']  # Updated in step_fn
            
            # Accumulate loss for epoch average
            epoch_loss += loss.item()
            steps_this_epoch += 1
            
            # Update step progress bar
            step_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Logging
            if global_step % cfg.training.log_freq == 0:
                avg_loss = epoch_loss / steps_this_epoch
                epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        # End of epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / steps_this_epoch
        epoch_losses.append(avg_epoch_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        
        # Save epoch metrics
        with open(os.path.join(log_dir, f"epoch_{epoch+1}_metrics.txt"), "w") as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Average Loss: {avg_epoch_loss:.6f}\n")
            f.write(f"Duration: {epoch_duration:.2f}s\n")
            f.write(f"Steps: {steps_this_epoch}\n")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch_{epoch+1}.pt'
        )
        torch.save(state, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch+1} to directory: {os.path.dirname(checkpoint_path)}")
        
        # Save EMA weights separately (often better for inference)
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        ema_path = os.path.join(
            checkpoint_dir,
            f'ema_weights_epoch_{epoch+1}.pt'
        )
        # torch.save(score_model.state_dict(), ema_path)
        ema.restore(score_model.parameters())
        print(f"Saved EMA weights for epoch {epoch+1} to directory: {os.path.dirname(ema_path)}")
    
    # End of training
    print("\nTraining completed!")
    print("Loss by epoch:")
    for i, loss in enumerate(epoch_losses):
        print(f"Epoch {i+1}: {loss:.6f}")
    
    # --- Save epoch loss data ---
    epoch_loss_path = os.path.join(log_dir, "epoch_losses.json")
    try:
        with open(epoch_loss_path, "w") as f:
            # Store as a list of [epoch_num, loss_value] for consistency with plotting
            epoch_loss_data = [[i+1, loss] for i, loss in enumerate(epoch_losses)]
            json.dump(epoch_loss_data, f)
        print(f"Saved epoch loss data to {epoch_loss_path}")
    except Exception as e:
        print(f"Error saving epoch loss data: {e}")
    # --- End saving epoch loss data ---

    # --- Plotting Epoch Loss Curve ---
    plot_path = os.path.join(output_dir, "epoch_loss_curve.png")
    try:
        if epoch_losses: # Check if there's data to plot
            epochs = range(1, len(epoch_losses) + 1)
            
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, epoch_losses, marker='o', linestyle='-', label='Average Training Loss per Epoch')
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title("Epoch vs. Average Training Loss")
            plt.xticks(epochs) # Ensure ticks for each epoch if not too many
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_path)
            plt.close() # Close the figure to free memory
            print(f"Saved epoch loss plot to {plot_path}")
        else:
            print("No epoch losses recorded, skipping plot generation.")
            
    except ImportError:
        print("Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'")
    except Exception as e:
        print(f"Error generating plot: {e}")
    # --- End Plotting Epoch Loss Curve ---

    # Save final metrics
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Final Loss: {epoch_losses[-1]:.6f}\n")
        f.write("Loss by epoch:\n")
        for i, loss in enumerate(epoch_losses):
            f.write(f"Epoch {i+1}: {loss:.6f}\n")

if __name__ == "__main__":
    main() 