import openai
import os
import time
import argparse
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def retrieve_batch_results(batch_id: str, output_dir: str = "batch_results"):
    """
    Retrieve results from a completed batch job and save to the batch_results directory.
    
    Args:
        batch_id: ID of the batch job
        output_dir: Directory to save the results (default: batch_results)
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename using the batch_id
    output_file = os.path.join(output_dir, f"{batch_id}.jsonl")
    
    # Get batch status
    batch = client.batches.retrieve(batch_id)
    print(f"Batch status: {batch.status}")
    
    # If batch failed or is still processing, handle accordingly
    if batch.status == "failed":
        print(f"Batch job failed")
    elif batch.status == "in_progress":
        print(f"Batch job is still processing. Try again later.")
        return
    
    # Check for output file (successful responses)
    output_file_id = batch.output_file_id
    if output_file_id:
        print(f"Found output file: {output_file_id}")
        # Download the output file
        response = client.files.content(output_file_id)
        with open(output_file, "wb") as f:
            f.write(response.read())
        print(f"Results downloaded to {output_file}")
    else:
        print("No output file found (no successful responses)")
    
    # Check for error file (failed responses)
    error_file_id = batch.error_file_id
    if error_file_id:
        print(f"Found error file: {error_file_id}")
        # Download the error file
        error_file = os.path.join(output_dir, f"{batch_id}_errors.jsonl")
        response = client.files.content(error_file_id)
        with open(error_file, "wb") as f:
            f.write(response.read())
        print(f"Error details downloaded to {error_file}")
        
        # Display error content
        try:
            with open(error_file, "r") as f:
                error_content = f.read()
                print("\nError details:")
                print(error_content)
        except Exception as e:
            print(f"Could not read error file: {e}")
    else:
        print("No error file found (no failed responses)")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Retrieve results from an OpenAI batch job")
    parser.add_argument("batch_id", type=str, help="ID of the batch job to retrieve (starting with 'batch_')")
    parser.add_argument("--output-dir", type=str, default="batch_results", 
                        help="Directory to save the results (default: batch_results)")
    args = parser.parse_args()
    
    # Retrieve batch results
    retrieve_batch_results(args.batch_id, args.output_dir)

if __name__ == "__main__":
    main()
