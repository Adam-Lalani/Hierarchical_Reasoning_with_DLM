import argparse
import pandas as pd
import ast
import logging
from tqdm import tqdm # Import tqdm

# Configure logging
# To see DEBUG messages, change level=logging.INFO to level=logging.DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
END_OF_TEXT_TOKEN = 50256
PADDING_TOKEN = 50257

# Primary target token sequence for '; Low-Level Reasoning ;'
TARGET_LOW_LEVEL_REASONING_TOKEN_SEQUENCE = [26, 7754, 12, 4971, 23219, 278, 2162]
# Fallback target token sequences
FALLBACK_TARGET_1_TOKEN_SEQUENCE = [20535, 12, 4971, 23219, 278] # 'Low-Level Reasoning'
FALLBACK_TARGET_2_TOKEN_SEQUENCE = [12, 4971, 23219] # part of 'Level Reasoning'

# The exact string to find for truncating full_sequence
TARGET_LOW_LEVEL_REASONING_STRING = "; Low-Level Reasoning ;"
MAX_LENGTH = 512  # Target length for the new high-level plan sequences

def find_subsequence_index(main_list, sub_list):
    """
    Finds the starting index of the first occurrence of sub_list in main_list.
    Returns the index or None if not found.
    """
    len_sub = len(sub_list)
    if not main_list or not sub_list: # Handle empty lists
        return None
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i:i+len_sub] == sub_list:
            return i
    return None

# Moved validate_output_dataframe function definition before main()
def validate_output_dataframe(df, expected_length):
    """
    Validates that 'input_ids' and 'attention_mask' in each row of the DataFrame,
    when parsed as lists, have the expected_length.
    """
    invalid_rows_count = 0
    # Use tqdm for validation progress as well, if df is large
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating rows"):
        # Validate input_ids
        try:
            input_ids_str = row.get('input_ids', '[]')
            input_ids_list = ast.literal_eval(input_ids_str)
            if not isinstance(input_ids_list, list):
                logging.error(f"Validation Error - Row {index} (DataFrame index): 'input_ids' is not a list after parsing. Value: '{input_ids_str[:100]}...'")
                invalid_rows_count += 1
                continue # Skip length check if not a list
            if len(input_ids_list) != expected_length:
                logging.error(f"Validation Error - Row {index} (DataFrame index): 'input_ids' has length {len(input_ids_list)}, expected {expected_length}. Value: '{str(input_ids_list)[:100]}...'")
                invalid_rows_count += 1
        except (ValueError, SyntaxError) as e:
            logging.error(f"Validation Error - Row {index} (DataFrame index): Could not parse 'input_ids'. Error: {e}. Value: '{row.get('input_ids', '')[:100]}...'")
            invalid_rows_count += 1
        
        # Validate attention_mask
        try:
            attention_mask_str = row.get('attention_mask', '[]')
            attention_mask_list = ast.literal_eval(attention_mask_str)
            if not isinstance(attention_mask_list, list):
                logging.error(f"Validation Error - Row {index} (DataFrame index): 'attention_mask' is not a list after parsing. Value: '{attention_mask_str[:100]}...'")
                invalid_rows_count += 1
                continue # Skip length check if not a list
            if len(attention_mask_list) != expected_length:
                logging.error(f"Validation Error - Row {index} (DataFrame index): 'attention_mask' has length {len(attention_mask_list)}, expected {expected_length}. Value: '{str(attention_mask_list)[:100]}...'")
                invalid_rows_count += 1
        except (ValueError, SyntaxError) as e:
            logging.error(f"Validation Error - Row {index} (DataFrame index): Could not parse 'attention_mask'. Error: {e}. Value: '{row.get('attention_mask', '')[:100]}...'")
            invalid_rows_count += 1
            
    if invalid_rows_count == 0:
        logging.info(f"Validation complete: All {len(df)} rows checked and conform to length {expected_length}.")
        return True
    else:
        logging.warning(f"Validation complete: Checked {len(df)} rows. Found {invalid_rows_count} row(s) with length discrepancies.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create a new dataset with original rows and high-level plan versions.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", required=True, help="Path to save the output CSV file.")
    args = parser.parse_args()

    logging.info(f"Loading data from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {args.input_csv}")
        return
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return

    processed_rows = []
    total_rows = len(df)
    hlp_created_count = 0
    hlp_skipped_token_not_found = 0
    hlp_skipped_parse_error = 0

    logging.info(f"Processing {total_rows} rows...")
    
    ten_percent_step = max(1, total_rows // 10) # Log at least once for small datasets, otherwise every 10%
    next_log_milestone_row_count = ten_percent_step

    for current_row_idx, (df_index, original_row) in enumerate(tqdm(df.iterrows(), total=total_rows, desc="Processing rows")):
        # Log progress at roughly 10% intervals
        # current_row_idx is 0-based, so add 1 for 1-based count
        if (current_row_idx + 1) >= next_log_milestone_row_count and (current_row_idx + 1) < total_rows:
            processed_percentage = ((current_row_idx + 1) / total_rows) * 100
            logging.info(f"Progress: Approximately {processed_percentage:.0f}% of rows processed ({current_row_idx + 1}/{total_rows}).")
            next_log_milestone_row_count += ten_percent_step

        processed_rows.append(original_row.to_dict())

        current_full_sequence = original_row.get('full_sequence', '')
        current_input_ids_str = original_row.get('input_ids', '[]')

        try:
            original_input_ids_list = ast.literal_eval(current_input_ids_str)
            if not isinstance(original_input_ids_list, list):
                raise ValueError("Parsed input_ids is not a list.")
        except (ValueError, SyntaxError) as e:
            logging.warning(f"Row {df_index} (DataFrame index): Could not parse input_ids. Skipping HLP version. Error: {e}. Data: '{current_input_ids_str[:100]}...'" )
            hlp_skipped_parse_error += 1
            continue

        truncation_token_idx = None
        search_targets = [
            TARGET_LOW_LEVEL_REASONING_TOKEN_SEQUENCE,
            FALLBACK_TARGET_1_TOKEN_SEQUENCE,
            FALLBACK_TARGET_2_TOKEN_SEQUENCE
        ]

        for target_seq in search_targets:
            truncation_token_idx = find_subsequence_index(original_input_ids_list, target_seq)
            if truncation_token_idx is not None:
                logging.debug(f"Row {df_index} (DataFrame index): Found target sequence {target_seq} at index {truncation_token_idx}.")
                break
        
        if truncation_token_idx is None:
            logging.warning(f"Row {df_index} (DataFrame index): None of the target token sequences found in input_ids. Skipping HLP version.")
            logging.debug(f"Row {df_index} (DataFrame index): Searched input_ids: {original_input_ids_list}")
            hlp_skipped_token_not_found += 1
            continue

        high_level_input_ids = original_input_ids_list[:truncation_token_idx]
        
        if len(high_level_input_ids) >= MAX_LENGTH:
            high_level_input_ids = high_level_input_ids[:MAX_LENGTH - 1]
        high_level_input_ids.append(END_OF_TEXT_TOKEN)

        num_padding_needed = MAX_LENGTH - len(high_level_input_ids)
        if num_padding_needed < 0: 
            logging.warning(f"Row {df_index} (DataFrame index): High-level plan plus EOS token exceeded MAX_LENGTH. Correcting. Length was {len(high_level_input_ids)}")
            high_level_input_ids = high_level_input_ids[:MAX_LENGTH - 1] + [END_OF_TEXT_TOKEN]
            num_padding_needed = 0
            
        high_level_input_ids.extend([PADDING_TOKEN] * num_padding_needed)

        content_len_for_mask = truncation_token_idx + 1 
        if content_len_for_mask > MAX_LENGTH:
            content_len_for_mask = MAX_LENGTH
        
        new_attention_mask = [1] * content_len_for_mask
        num_attn_padding = MAX_LENGTH - len(new_attention_mask)
        if num_attn_padding < 0: 
            new_attention_mask = new_attention_mask[:MAX_LENGTH]
            num_attn_padding = 0
        new_attention_mask.extend([0] * num_attn_padding)
        
        high_level_full_sequence = current_full_sequence 
        if isinstance(current_full_sequence, str):
            text_truncation_idx = current_full_sequence.find(TARGET_LOW_LEVEL_REASONING_STRING)
            if text_truncation_idx != -1:
                high_level_full_sequence = current_full_sequence[:text_truncation_idx]
            else:
                logging.warning(f"Row {df_index} (DataFrame index): Target string '{TARGET_LOW_LEVEL_REASONING_STRING}' not found in full_sequence. Using original for HLP.")
        else:
            logging.warning(f"Row {df_index} (DataFrame index): full_sequence is not a string or is missing. Using original for HLP.")

        new_hlp_row = {
            'full_sequence': high_level_full_sequence,
            'input_ids': str(high_level_input_ids),
            'attention_mask': str(new_attention_mask)
        }
        
        for col_name, col_value in original_row.items():
            if col_name not in new_hlp_row: 
                new_hlp_row[col_name] = col_value
        
        processed_rows.append(new_hlp_row)
        hlp_created_count += 1

    logging.info(f"Finished processing {total_rows} original rows.")
    logging.info(f"Successfully created {hlp_created_count} high-level plan (HLP) rows.")
    if total_rows > 0:
        success_percentage = (hlp_created_count / total_rows) * 100
        logging.info(f"HLP creation success rate: {success_percentage:.2f}% of original rows resulted in an HLP version.")
    logging.info(f"HLP rows skipped due to token sequence not found: {hlp_skipped_token_not_found}")
    logging.info(f"HLP rows skipped due to input_ids parse error: {hlp_skipped_parse_error}")

    output_df = pd.DataFrame(processed_rows)

    logging.info("Validating the structure of the generated DataFrame...")
    validation_passed = validate_output_dataframe(output_df, MAX_LENGTH)
    if validation_passed:
        logging.info("DataFrame validation successful: All input_ids and attention_mask fields have the correct length.")
    else:
        logging.warning("DataFrame validation failed: Some rows have incorrect lengths for input_ids or attention_mask. Check logs above.")

    logging.info(f"Saving {len(output_df)} total rows to: {args.output_csv}")
    try:
        output_df.to_csv(args.output_csv, index=False)
        logging.info("Successfully saved output CSV.")
    except Exception as e:
        logging.error(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    main() 