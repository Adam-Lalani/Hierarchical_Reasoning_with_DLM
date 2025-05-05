import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast
import math
import time

# Add parent directory to path to import from original codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SEDD
from model.ema import ExponentialMovingAverage
import utils
import noise_lib
import graph_lib
import losses

from hierarchical_dataset import get_math_dataloaders
from custom_losses import get_hierarchical_step_fn

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    # Update config for our specific case - A100 specific settings
    cfg.ngpus = 1  # Only using 1 A100
    cfg.data.train = "data_/train_set.csv"
    cfg.data.valid = "data_/val_set.csv"
    cfg.training.batch_size = 2  # A100 has 40-80GB memory, can handle larger batches
    cfg.training.n_iters = 10000  # Will be overridden by epoch-based training
    cfg.optim.lr = 2e-5  # Smaller learning rate for fine-tuning
    cfg.optim.warmup = 100  # Shorter warmup for fine-tuning
    cfg.training.log_freq = 10  # More frequent logging
    
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
    num_epochs = 2  # Train for 2 epochs
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
    
    for epoch in range(num_epochs):
        state['epoch'] = epoch
        epoch_start_time = time.time()
        epoch_loss = 0.0
        steps_this_epoch = 0
        
        # Reset iterator for each epoch
        train_iter = iter(train_loader)
        
        for step in range(steps_per_epoch):
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
            
            # Logging
            if global_step % cfg.training.log_freq == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, "
                      f"Global Step {global_step}, Loss: {loss.item():.4f}")
        
        # End of epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / steps_this_epoch
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s")
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
        print(f"Saved checkpoint for epoch {epoch+1}")
        
        # Save EMA weights separately (often better for inference)
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        ema_path = os.path.join(
            checkpoint_dir,
            f'ema_weights_epoch_{epoch+1}.pt'
        )
        torch.save(score_model.state_dict(), ema_path)
        ema.restore(score_model.parameters())
        print(f"Saved EMA weights for epoch {epoch+1}")
    
    # End of training
    print("Training completed!")
    print("Loss by epoch:")
    for i, loss in enumerate(epoch_losses):
        print(f"Epoch {i+1}: {loss:.6f}")
    
    # Save final metrics
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Final Loss: {epoch_losses[-1]:.6f}\n")
        f.write("Loss by epoch:\n")
        for i, loss in enumerate(epoch_losses):
            f.write(f"Epoch {i+1}: {loss:.6f}\n")

if __name__ == "__main__":
    main() 