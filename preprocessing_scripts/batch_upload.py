import os
import json
import time
import argparse
import math
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any, Tuple
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

def load_metamath_dataset(limit=None):
    """
    Load the MetaMathQA dataset from Hugging Face.
    
    Args:
        limit: Optional limit on the number of examples to load
    
    Returns:
        The MetaMathQA dataset
    """
    print("Loading MetaMathQA dataset from Hugging Face...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Example features: {dataset.features}")
    print(f"First example: {dataset[0]}")
    
    return dataset

def process_dataset_chunk(dataset, start_idx: int, end_idx: int, chunk_num: int, 
                         output_dir: str, model: str) -> Tuple[str, str]:
    """
    Process a chunk of the dataset and create a batch job for it.
    
    Args:
        dataset: The dataset to process
        start_idx: Start index of the chunk
        end_idx: End index of the chunk
        chunk_num: Chunk number for identification
        output_dir: Directory to save output files
        model: OpenAI model to use
        
    Returns:
        Tuple of (batch_id, output_file)
    """
    # Create a subset of the dataset for this chunk
    chunk_dataset = dataset.select(range(start_idx, end_idx))
    
    # Create an output file for this chunk with indices in the filename
    output_file = os.path.join(output_dir, f"chunk_{start_idx}_{end_idx}.jsonl")
    
    # Create JSONL file for this chunk
    print(f"Creating batch file for chunk {chunk_num} with {len(chunk_dataset)} requests...")
    create_batch_jsonl_file(chunk_dataset, output_file, model, start_idx)
    
    # Upload file to OpenAI
    print(f"Uploading chunk {chunk_num} to OpenAI...")
    file_id = upload_file_to_openai(output_file)
    
    # Create batch job
    print(f"Creating batch job for chunk {chunk_num}...")
    batch = create_batch_job(file_id)
    
    return batch.id, output_file

def create_batch_jsonl_file(dataset, output_file: str, model: str = "gpt-4o-mini-2024-07-18", 
                           start_idx: int = 0) -> None:
    """
    Create a JSONL file for OpenAI's Batch API based on the MetaMathQA dataset.
    
    Args:
        dataset: Hugging Face dataset with question and response fields
        output_file: Path to save the JSONL file
        model: OpenAI model to use for the batch
        start_idx: Starting index for this chunk (for global indexing)
    """
    batch_requests = []
    
    system_prompt = """
I am going to give you a math question, and I want you to reformat the question to clearly distinguish the high-level planning steps that you'll take in order to solve the problem vs. the details of the reasoning that you must enact for each step. Since different questions require a different number of steps, you may use K steps, where K can be any positive number. Here is the outline of the format that you must follow for every question:

"Step 1: [insert description of Step 1] ; [insert details of reasoning for Step 1]
Step 2: [insert description of Step 2] ; [insert details of reasoning for Step 2]
... 
Step K: [insert description of Step K] ; [insert details of reasoning for Step K]
Conclusion: The answer is [Insert answer in numerical form]"
"""

    
    # Create batch request object for each item in the dataset
    for i, item in enumerate(dataset):
        # Calculate global index
        global_idx = start_idx + i

        user_prompt = f"""
            Question:
            {item['original_question']}

            Answer:
            {item['response']}
            """
        
        
        # Format messages in OpenAI chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Create a request object for this item using the required batch API format
        request = {
            "custom_id": f"req_{global_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0.00,
                "max_tokens": 1024,
            }
        }
        
        # Add to batch requests
        batch_requests.append(request)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write to JSONL file (one JSON object per line)
    with open(output_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created batch file {output_file} with {len(batch_requests)} requests")

def upload_file_to_openai(file_path: str) -> str:
    """
    Upload a file to OpenAI for batch processing.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        File ID from OpenAI
    """
    # Set up the OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Upload the file
    with open(file_path, 'rb') as file:
        response = client.files.create(
            file=file,
            purpose="batch"
        )
    
    print(f"Uploaded file {file_path} to OpenAI with ID: {response.id}")
    return response.id

def create_batch_job(file_id: str) -> Dict[str, Any]:
    """
    Create a batch job on OpenAI with the uploaded file.
    
    Args:
        file_id: ID of the file uploaded to OpenAI
        
    Returns:
        Batch job details
    """
    # Set up the OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create the batch
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Created batch job with ID: {batch.id}")
    return batch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create and submit batch requests to OpenAI")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to process (for testing)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="OpenAI model to use")
    parser.add_argument("--output-dir", type=str, default="batches", help="Directory to save output files")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of examples per batch")
    parser.add_argument("--dry-run", action="store_true", help="Create JSONL files but don't submit to OpenAI")
    parser.add_argument("--start-idx", type=int, default=None, help="Starting index for processing a slice of the dataset")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending index for processing a slice of the dataset")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MetaMathQA dataset from Hugging Face
    dataset = load_metamath_dataset(limit=args.limit)
    
    # Determine the start and end indices
    total_examples = len(dataset)
    
    start_idx = 0 if args.start_idx is None else min(args.start_idx, total_examples)
    end_idx = total_examples if args.end_idx is None else min(args.end_idx, total_examples)
    
    # Validate indices
    if start_idx >= end_idx:
        print(f"Error: start_idx ({start_idx}) must be less than end_idx ({end_idx})")
        return
    
    # Calculate number of chunks
    examples_to_process = end_idx - start_idx
    num_chunks = math.ceil(examples_to_process / args.chunk_size)
    print(f"Processing {examples_to_process} examples from indices {start_idx} to {end_idx-1} in {num_chunks} chunks of {args.chunk_size}")
    
    # Keep track of batch jobs
    batch_jobs = []
    
    # Process each chunk
    for chunk_num in range(num_chunks):
        chunk_start_idx = start_idx + (chunk_num * args.chunk_size)
        chunk_end_idx = min(chunk_start_idx + args.chunk_size, end_idx)
        
        print(f"\nProcessing chunk {chunk_num+1}/{num_chunks} (examples {chunk_start_idx}-{chunk_end_idx-1})...")
        
        if args.dry_run:
            # Create JSONL file only
            output_file = os.path.join(args.output_dir, f"chunk_{chunk_start_idx}_{chunk_end_idx}.jsonl")
            create_batch_jsonl_file(dataset.select(range(chunk_start_idx, chunk_end_idx)), output_file, args.model, chunk_start_idx)
            print(f"Dry run: created file {output_file}")
        else:
            # Process the chunk and create a batch job
            batch_id, output_file = process_dataset_chunk(
                dataset, chunk_start_idx, chunk_end_idx, chunk_num, args.output_dir, args.model
            )
            batch_jobs.append((batch_id, output_file))
    
    # Write batch job info to a file
    if not args.dry_run and batch_jobs:
        with open(os.path.join(args.output_dir, "batch_jobs.txt"), "w") as f:
            f.write("Batch Jobs:\n")
            for idx, (batch_id, output_file) in enumerate(batch_jobs):
                f.write(f"Chunk {idx}: {batch_id} - {output_file}\n")
        
        print("\nBatch jobs created successfully!")
        print(f"Batch job IDs saved to {os.path.join(args.output_dir, 'batch_jobs.txt')}")
        print("You can check the status of your batch jobs using the OpenAI API or dashboard.")

if __name__ == "__main__":
    main() 