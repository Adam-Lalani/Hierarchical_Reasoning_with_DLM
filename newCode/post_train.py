import datetime
import os
import os.path
import gc
from itertools import chain
import functools # Added for partial

import numpy as np
import pandas as pd # Added for CSV loading
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader # Added
from torch.utils.data.distributed import DistributedSampler # Added

# Removed data import, added custom dataset/collate imports
from newCode.temp_sampling_dataset_builder import JointGSM8kDataset
from newCode.collate import joint_collate_fn
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    # TODO: Load pre-trained weights here if desired before wrapping with DDP
    # Example: state_dict = torch.load(cfg.model.pretrained_path, map_location=device)
    #          score_model.load_state_dict(state_dict) 
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], find_unused_parameters=False) # Set find_unused_parameters=False if possible

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    # If noise schedule parameters are not trained, DDP wrapper might not be needed
    # noise = DDP(noise, device_ids=[rank], find_unused_parameters=False) 
    sampling_eps = cfg.model.sampling_eps # Use config value

    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters())) # Only optimize score_model params
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp) # Enable AMP based on config
    mprint(f"Scaler: {scaler}")
    # Removed noise from state as we are not optimizing it
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, ema=ema, step=0) 

    # load in state (checkpoint) - Loads optimizer, scaler, model, ema, step
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # --- Modified Data Loading ---
    # load in tokenizer
    # Use the specific tokenizer potentially defined in config
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.model.tokenizer_name) 
    # Ensure mask token is added by the dataset
    mask_token = cfg.data.mask_token 
    # Define pad token id (use eos_token_id if tokenizer doesn't have a specific pad_token_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Load data from CSV
    df_train = pd.read_csv(cfg.data.train_csv)
    # Optional: Load validation data
    df_valid = pd.read_csv(cfg.data.valid_csv) if cfg.data.valid_csv else None

    # Convert DataFrame to list of dicts
    train_data_list = df_train.to_dict(orient='records')
    valid_data_list = df_valid.to_dict(orient='records') if df_valid is not None else None

    # Create Datasets
    train_dataset = JointGSM8kDataset(
        train_data_list, 
        tokenizer, 
        max_length=cfg.data.max_length, 
        mask_token=mask_token
    )
    if valid_data_list:
        eval_dataset = JointGSM8kDataset(
            valid_data_list, 
            tokenizer, 
            max_length=cfg.data.max_length, 
            mask_token=mask_token
        )
    else:
        # Use a subset of training data for evaluation if no validation set provided
        mprint("No validation CSV provided, using a subset of training data for evaluation.")
        eval_indices = list(range(min(len(train_dataset), cfg.training.eval_subset_size))) # Use first N samples
        eval_dataset = torch.utils.data.Subset(train_dataset, eval_indices)


    # Create Distributed Samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Create DataLoaders
    # Use partial to pass pad_token_id to collate_fn
    custom_collate_fn = functools.partial(joint_collate_fn, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size // world_size, # Per-GPU batch size
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=True, # Important for balanced batches
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval.batch_size // world_size, # Per-GPU batch size
        sampler=eval_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=False, # Keep all validation samples
    )

    mprint(f"Train dataset size: {len(train_dataset)}, Train loader steps per epoch: {len(train_loader)}")
    mprint(f"Eval dataset size: {len(eval_dataset)}, Eval loader steps per epoch: {len(eval_loader)}")

    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)
    # --- End Modified Data Loading ---


    # --- Modified Step Function Creation ---
    optimize_fn = losses.optimization_manager(cfg)
    
    # Determine the token ID for weighting the final answer
    final_answer_token_id = None
    if cfg.training.final_answer_token:
        # Try to get the token ID for the specified marker token
        tokens = tokenizer.encode(cfg.training.final_answer_token)
        if len(tokens) == 1:
            final_answer_token_id = tokens[0]
            mprint(f"Using token ID {final_answer_token_id} ('{cfg.training.final_answer_token}') to mark start of final answer for weighting.")
        else:
            mprint(f"Warning: final_answer_token '{cfg.training.final_answer_token}' is tokenized into multiple IDs ({tokens}). Cannot use for weighting. Set final_answer_weight to 1.0 or use a single-token marker.")
            cfg.training.final_answer_weight = 1.0 # Disable weighting if token is invalid
    
    # Pass weighting parameters to get_step_fn
    train_step_fn = losses.get_step_fn(
        noise, graph, True, optimize_fn, cfg.training.accum,
        final_answer_weight=cfg.training.final_answer_weight,
        final_answer_token_id=final_answer_token_id,
        pad_token_id=pad_token_id
    )
    eval_step_fn = losses.get_step_fn(
        noise, graph, False, optimize_fn, cfg.training.accum, # Note: Accumulation usually not used in eval
        final_answer_weight=cfg.training.final_answer_weight, 
        final_answer_token_id=final_answer_token_id,
        pad_token_id=pad_token_id
    )
    # --- End Modified Step Function Creation ---


    if cfg.training.snapshot_sampling:
        # Adjust sampling shape for per-GPU batch size
        sampling_shape = (cfg.training.batch_size // world_size, cfg.data.max_length) 
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")
    epoch = 0

    while state['step'] < num_train_steps + 1:
        # Set epoch for DistributedSampler to ensure shuffling
        train_sampler.set_epoch(epoch)
        
        # --- Modified Training Loop ---
        for batch_data in train_loader:
            if state['step'] > num_train_steps:
                break # Exit if max steps reached within the epoch

            step = state['step'] 
            # The collate function returns dict: {"input_ids", "attention_mask", "labels"}
            # For SEDD, the 'batch' for loss calculation is the target sequence ('labels')
            batch_labels = batch_data['labels'].to(device) 
            
            loss = train_step_fn(state, batch_labels) # Pass labels as the batch

            # Check if an optimizer step actually happened (depends on accumulation)
            if step != state['step']: 
                current_step = state['step'] # Use the updated step count

                if current_step % cfg.training.log_freq == 0:
                    # loss is already reduced/averaged within step_fn if accumulation happened
                    # If using DDP, gradients are averaged, loss might need explicit averaging if not accumulated fully
                    log_loss = loss # Use the loss returned by step_fn
                    # Optional: Gather loss from all ranks if needed (e.g., if accum=1)
                    # dist.all_reduce(log_loss) 
                    # log_loss /= world_size 
                    mprint("step: %d, training_loss: %.5e" % (current_step, log_loss.item()))
                
                # Preemption checkpoint saving
                if current_step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                    utils.save_checkpoint(checkpoint_meta_dir, state)

                # Evaluation
                if current_step % cfg.training.eval_freq == 0:
                    eval_sampler.set_epoch(epoch) # Set epoch for eval sampler too
                    total_eval_loss = 0.0
                    eval_steps = 0
                    for eval_batch_data in eval_loader:
                        eval_batch_labels = eval_batch_data['labels'].to(device)
                        eval_loss = eval_step_fn(state, eval_batch_labels)
                        # Gather loss across ranks for accurate evaluation
                        dist.all_reduce(eval_loss)
                        eval_loss /= world_size
                        total_eval_loss += eval_loss.item()
                        eval_steps += 1
                        if eval_steps >= cfg.training.eval_steps_limit: # Limit eval steps
                           break 
                    
                    average_eval_loss = total_eval_loss / eval_steps if eval_steps > 0 else 0.0
                    mprint("step: %d, evaluation_loss: %.5e" % (current_step, average_eval_loss))
                    # Reset eval iterator if needed (or rely on DataLoader recreation)
                    # eval_iter = iter(eval_loader) 


                # Snapshot saving and sampling
                if current_step > 0 and (current_step % cfg.training.snapshot_freq == 0 or current_step == num_train_steps):
                    # Save the checkpoint.
                    save_step = current_step // cfg.training.snapshot_freq
                    if rank == 0:
                        utils.save_checkpoint(os.path.join(
                            checkpoint_dir, f'checkpoint_{current_step}.pth'), state) # Use step in filename

                    # Generate and save samples
                    if cfg.training.snapshot_sampling:
                        mprint(f"Generating text at step: {current_step}")

                        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(current_step))
                        if rank == 0: utils.makedirs(this_sample_dir)
                        dist.barrier() # Ensure directory exists before writing

                        ema.store(score_model.parameters())
                        ema.copy_to(score_model.parameters())
                        sample = sampling_fn(score_model.module) # Use model.module with DDP
                        ema.restore(score_model.parameters())

                        # Decode requires handling padding tokens
                        sentences = tokenizer.batch_decode(sample, skip_special_tokens=True) 
                        
                        file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                        with open(file_name, 'w') as file:
                            for sentence in sentences:
                                file.write(sentence + "\n")
                                file.write("============================================================================================\n")
                        
                        # Optional Perplexity calculation - Adapt if needed
                        # ... (perplexity code might need adjustment for padding/batching) ...

                        dist.barrier() # Ensure all ranks finish sampling before next step
        # --- End Modified Training Loop ---
        
        epoch += 1 # Increment epoch after iterating through train_loader
        # Break if we have reached the target number of steps regardless of epochs
        if state['step'] > num_train_steps:
             break

    mprint(f"Training finished after {state['step']} steps.")

# Add a main guard and argument parsing similar to run_train.py if this script is meant to be run directly
# Example:
# if __name__ == "__main__":
#     # Parse arguments (config path, etc.)
#     # Load config using utils.load_config
#     # Set up multiprocessing (port, world_size)
#     # Call run_multiprocess
#     pass 