import os
import json
import time
import argparse
import math
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any, Tuple, Optional

# Import functions from batch_upload.py - fixing the import path
from batch_upload import (
    load_metamath_dataset, 
    create_batch_jsonl_file, 
    upload_file_to_openai, 
    create_batch_job
)

# Load environment variables from .env file
load_dotenv()

def check_batch_status(batch_id: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Check the status of a batch job.
    
    Args:
        batch_id: ID of the batch job
        
    Returns:
        Tuple of (status, output_file_id, error_file_id)
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get batch status
    batch = client.batches.retrieve(batch_id)
    
    return batch.status, batch.output_file_id, batch.error_file_id

def download_batch_results(batch_id: str, output_dir: str = "batch_results") -> bool:
    """
    Download results from a completed batch job.
    
    Args:
        batch_id: ID of the batch job
        output_dir: Directory to save the results
        
    Returns:
        True if download was successful and no errors, False otherwise
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get batch status
    status, output_file_id, error_file_id = check_batch_status(batch_id)
    
    if status != "completed":
        print(f"Batch is not completed. Current status: {status}")
        return False
    
    # Check for output file (successful responses)
    if output_file_id:
        output_file = os.path.join(output_dir, f"{batch_id}.jsonl")
        print(f"Found output file: {output_file_id}")
        # Download the output file
        response = client.files.content(output_file_id)
        with open(output_file, "wb") as f:
            f.write(response.read())
        print(f"Results downloaded to {output_file}")
    else:
        print("No output file found (no successful responses)")
        return False
    
    # Check for error file (failed responses)
    if error_file_id:
        error_file = os.path.join(output_dir, f"{batch_id}_errors.jsonl")
        print(f"Found error file: {error_file_id}")
        # Download the error file
        response = client.files.content(error_file_id)
        with open(error_file, "wb") as f:
            f.write(response.read())
        print(f"Error details downloaded to {error_file}")
        
        # Error file exists, but we'll still count it as success for moving to next batch
        # Just note the errors
        with open(error_file, "r") as f:
            error_lines = f.readlines()
            print(f"⚠️ Batch completed with {len(error_lines)} errors")
        
        return True  # We'll continue even with some errors
    
    # No errors found
    print("✅ Batch completed successfully with no errors")
    return True

def process_and_wait_for_batch(dataset, chunk_start_idx: int, chunk_end_idx: int, 
                             chunk_num: int, output_dir: str, results_dir: str,
                             model: str, check_interval_minutes: int = 15) -> bool:
    """
    Process a chunk of the dataset, create a batch job, and wait for it to complete.
    
    Args:
        dataset: The dataset to process
        chunk_start_idx: Start index of the chunk
        chunk_end_idx: End index of the chunk
        chunk_num: Chunk number for identification
        output_dir: Directory to save output files
        results_dir: Directory to save result files
        model: OpenAI model to use
        check_interval_minutes: How often to check batch status (in minutes)
        
    Returns:
        True if batch completed successfully, False otherwise
    """
    # Create a subset of the dataset for this chunk
    chunk_dataset = dataset.select(range(chunk_start_idx, chunk_end_idx))
    
    # Create an output file for this chunk with indices in the filename
    output_file = os.path.join(output_dir, f"chunk_{chunk_start_idx}_{chunk_end_idx}.jsonl")
    
    # Create JSONL file for this chunk
    print(f"Creating batch file for chunk {chunk_num} with {len(chunk_dataset)} requests...")
    create_batch_jsonl_file(chunk_dataset, output_file, model, chunk_start_idx)
    
    # Upload file to OpenAI
    print(f"Uploading chunk {chunk_num} to OpenAI...")
    file_id = upload_file_to_openai(output_file)
    
    # Create batch job
    print(f"Creating batch job for chunk {chunk_num}...")
    batch = create_batch_job(file_id)
    batch_id = batch.id
    
    # Wait for batch to complete
    print(f"Waiting for batch {batch_id} to complete...")
    check_interval_seconds = check_interval_minutes * 60
    
    while True:
        # Get current time for logging
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Check batch status
        status, _, _ = check_batch_status(batch_id)
        print(f"[{current_time}] Batch {batch_id} status: {status}")
        
        if status == "completed":
            # Batch is complete, download results
            success = download_batch_results(batch_id, results_dir)
            return success
        elif status in ["failed", "cancelled", "expired"]:
            # Batch failed
            print(f"⚠️ Batch {batch_id} {status}. Cannot proceed with sequential processing.")
            return False
        
        # Wait before checking again
        print(f"Waiting {check_interval_minutes} minutes before checking again...")
        time.sleep(check_interval_seconds)

def sequential_batch_processing(start_idx: int = 0, end_idx: Optional[int] = None, 
                              chunk_size: int = 10000, model: str = "gpt-4o-mini-2024-07-18",
                              output_dir: str = "batches", results_dir: str = "batch_results",
                              check_interval_minutes: int = 15, limit: Optional[int] = None):
    """
    Process the dataset in sequential batches, waiting for each to complete.
    
    Args:
        start_idx: Starting index for processing
        end_idx: Ending index for processing (None = to the end)
        chunk_size: Number of examples per chunk
        model: OpenAI model to use
        output_dir: Directory to save batch files
        results_dir: Directory to save results
        check_interval_minutes: How often to check batch status (in minutes)
        limit: Optional limit on the number of examples to load
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load MetaMathQA dataset from Hugging Face
    dataset = load_metamath_dataset(limit=limit)
    
    # Determine the end index if not specified
    total_examples = len(dataset)
    if end_idx is None:
        end_idx = total_examples
    else:
        end_idx = min(end_idx, total_examples)
    
    # Validate indices
    if start_idx >= end_idx:
        print(f"Error: start_idx ({start_idx}) must be less than end_idx ({end_idx})")
        return
    
    # Calculate number of chunks
    examples_to_process = end_idx - start_idx
    num_chunks = math.ceil(examples_to_process / chunk_size)
    print(f"Processing {examples_to_process} examples from indices {start_idx} to {end_idx-1}")
    print(f"Will process in {num_chunks} sequential chunks of {chunk_size}")
    
    # Log start time
    start_time = time.time()
    
    # Process each chunk sequentially
    for chunk_num in range(num_chunks):
        chunk_start_idx = start_idx + (chunk_num * chunk_size)
        chunk_end_idx = min(chunk_start_idx + chunk_size, end_idx)
        
        print(f"\n{'='*80}")
        print(f"PROCESSING CHUNK {chunk_num+1}/{num_chunks}")
        print(f"Indices {chunk_start_idx} to {chunk_end_idx-1}")
        print(f"{'='*80}")
        
        # Process this chunk and wait for it to complete
        success = process_and_wait_for_batch(
            dataset, chunk_start_idx, chunk_end_idx, chunk_num, 
            output_dir, results_dir, model, check_interval_minutes
        )
        
        if not success:
            print("\n❌ Error processing batch. Stopping sequential processing.")
            break
        
        print(f"\n✅ Chunk {chunk_num+1}/{num_chunks} completed successfully")
        
        # Log progress
        elapsed_time = time.time() - start_time
        avg_time_per_chunk = elapsed_time / (chunk_num + 1)
        remaining_chunks = num_chunks - (chunk_num + 1)
        est_remaining_time = avg_time_per_chunk * remaining_chunks
        
        print(f"\nProgress: {chunk_num+1}/{num_chunks} chunks completed ({(chunk_num+1)/num_chunks*100:.1f}%)")
        print(f"Elapsed time: {elapsed_time/3600:.1f} hours")
        if remaining_chunks > 0:
            print(f"Estimated time remaining: {est_remaining_time/3600:.1f} hours")
    
    # Log completion
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time/3600:.1f} hours")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process dataset in sequential batches, waiting for each to complete"
    )
    parser.add_argument("--start-idx", type=int, default=0, 
                        help="Starting index for processing")
    parser.add_argument("--end-idx", type=int, default=None, 
                        help="Ending index for processing (default: to the end)")
    parser.add_argument("--chunk-size", type=int, default=10000, 
                        help="Number of examples per batch chunk (default: 10000)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", 
                        help="OpenAI model to use")
    parser.add_argument("--output-dir", type=str, default="batches", 
                        help="Directory to save batch files")
    parser.add_argument("--results-dir", type=str, default="batch_results", 
                        help="Directory to save results")
    parser.add_argument("--check-interval", type=int, default=15, 
                        help="How often to check batch status (in minutes)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of examples to process (for testing)")
    args = parser.parse_args()
    
    # Run sequential batch processing
    sequential_batch_processing(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        chunk_size=args.chunk_size,
        model=args.model,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        check_interval_minutes=args.check_interval,
        limit=args.limit
    )

if __name__ == "__main__":
    main() 