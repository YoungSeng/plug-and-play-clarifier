# -*- coding: utf-8 -*-
"""
This script analyzes conversational data from a JSONL file using a specified
Language Model (LLM). It is designed to be run from the command line,
accepting paths for input data, the model, and output directory as arguments.

The core task is to process each assistant's turn in a conversation,
disentangling the questions asked and extracting any options provided.

How to Run:
-----------
python your_script_name.py \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --input_file "/path/to/your/eval_output_model.jsonl" \
    --output_dir "./output_results"
"""

import os
import copy
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# 1. ARGUMENT PARSING
# ==============================================================================

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze conversational data using an LLM to disentangle queries and options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to a local model or a Hugging Face model ID for analysis. "
             "Example (HF): 'Qwen/Qwen2.5-7B-Instruct'. "
             "Example (Local): '/path/to/local/model'."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing conversational data. "
             "Example: './output/eval_output_Qwen2.5-7B-Instruct.jsonl'"
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory where the output file will be saved."
    )
    parser.add_argument(
        "--human_labels_path",
        type=str,
        default="./data/test_data_labels.jsonl",
        help="Path to human labels for potential future evaluation (currently unused in this script)."
    )

    return parser.parse_args()


# ==============================================================================
# 2. MODEL INITIALIZATION AND INFERENCE FUNCTIONS
#    (These functions remain the same as before)
# ==============================================================================

def init_qwen_model(model_path: str):
    """Initializes and loads the Qwen model and tokenizer."""
    print(f"Loading model and tokenizer from: {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✅ Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        raise RuntimeError("Failed to initialize Qwen model.") from e


def qwen_chat_completion(model, tokenizer, messages: list) -> str:
    """Performs inference using a loaded Qwen model."""
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    input_token_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_token_len:].tolist()

    try:
        think_end_index = len(output_ids) - output_ids[::-1].index(151668)  # 151668 is '</think>'
        response_ids = output_ids[think_end_index:]
    except ValueError:
        response_ids = output_ids

    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response


# ==============================================================================
# 3. PROMPT DEFINITION
# ==============================================================================

SYS_PROMPT = """You are a helpful assistant in understanding and analyzing user's queries. You are tasked with two missions:
1. What queries does the user make, disentangle them and list them line by line. You should identify the specific querying aspects (instead of general ones) and disentangle them.
2. For each query, what options does the user provide, list them after the query and "--" one by one, separated by a comma. If no options are parsed, leave the space after "--" empty.
Each time, the user will present a query. Please respond following the format given in the chat history. First, provide your thought, then present the response in a structured format."""

FEW_SHOT_EXAMPLES = [
    {'role': 'system', 'content': SYS_PROMPT},
    {'role': 'user',
     'content': "Can you tell me a bit more about the novel ABC? What is the main plot or theme of the novel?"},
    {'role': 'assistant',
     'content': "Thought:\nThe user is asking for two specific aspects of the novel ABC but does not provide the option for any query.\nResponse:\nWhat's the main plot of the novel ABC? --\nWhat's the theme of the novel ABC? --"},
    {'role': 'user',
     'content': "Awesome, there are usually some cool ones in China! How much are you looking to spend on attending the festival? Would it be under $500, between $500 - $1500, or above $1500?"},
    {'role': 'assistant',
     'content': "Thought:\nThe user only asks one query with three options.\nResponse:\nHow much are you looking to spend on attending the festival? -- under $500, between $500 - $1500, above $1500"},
    {'role': 'user',
     'content': "Can you please provide more details about the music festival you're looking for? For example, what type of music do you prefer (e.g. rock, pop, electronic)? Is it a local festival or one that's happening in a different city? Are there any specific dates or venues you're interested in?"},
    {'role': 'assistant',
     'content': "Thought:\nThe user asks for the type of music, where it is held, dates, and venues. There are in total 4 aspects that should be disentangled.\nResponse:\nWhat type of music do you prefer? -- rock, pop, electronic\nWhere do you prefer the music festival to be located? -- locally, in a different city\nWhat's the date of the music festival? --\nWhat's the venue of the music festival? --"},
]


# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================

def main():
    """Main function to run the data processing pipeline."""
    args = get_args()

    # --- Print configuration for user confirmation ---
    print("--- Configuration ---")
    print(f"  Model for Analysis: {args.model_name_or_path}")
    print(f"  Input Data File:    {args.input_file}")
    print(f"  Output Directory:   {args.output_dir}")
    print(f"  Human Labels Path:  {args.human_labels_path} (note: currently unused)")
    print("---------------------\n")

    # --- Step 1: Construct Paths and Load Data ---
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input data file not found at {args.input_file}")
        return

    # Automatically determine output filename based on input filename
    base_filename = os.path.basename(args.input_file)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_split{ext}"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Loading interaction data from: {args.input_file}")
    interaction_dataset = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            interaction_dataset.append(json.loads(line))
    print(f"Loaded {len(interaction_dataset)} interaction records.")

    # --- Step 2: Initialize the Analysis Model ---
    try:
        qwen_model, qwen_tokenizer = init_qwen_model(args.model_name_or_path)
    except RuntimeError:
        print("Exiting due to model initialization failure.")
        return

    # --- Step 3: Process Each Interaction Record ---
    num_records = len(interaction_dataset)
    print(f"\nProcessing {num_records} interaction records...")

    for i, record in enumerate(interaction_dataset):
        print(f"\n=========== Processing Record {i + 1}/{num_records} ===========")
        num_turns_to_process = len(record['actions']) // 2 - 1

        if num_turns_to_process <= 0:
            print("No assistant turns to process in this record, skipping.")
            record['query_options_list'] = []
            continue

        task_query_options_list = []
        print(f"Analyzing {num_turns_to_process} assistant turn(s) in this record.")
        for j in tqdm(range(num_turns_to_process), desc=f"Record {i + 1}"):
            turn_index = 2 * j + 1
            assistant_turn_content = record['actions'][turn_index]['content']

            messages = copy.deepcopy(FEW_SHOT_EXAMPLES)
            messages.append({"role": "user", "content": assistant_turn_content})

            try:
                raw_resp = qwen_chat_completion(qwen_model, qwen_tokenizer, messages)

                if "Response:" in raw_resp:
                    response_part = raw_resp.split("Response:", 1)[-1].strip()
                else:
                    print(
                        f"\n⚠️ Warning: 'Response:' marker not found for record {i + 1}, turn {j + 1}. Parsing raw output.")
                    response_part = raw_resp

                resp_lines = response_part.split('\n')
                turn_query_options = []
                for line in resp_lines:
                    line = line.strip()
                    if not line:
                        continue

                    if '--' in line:
                        query, options_str = line.split('--', 1)
                        turn_query_options.append({
                            "query": query.strip(),
                            "options": [opt.strip() for opt in
                                        options_str.strip().split(',')] if options_str.strip() else []
                        })
                    else:
                        print(f"\n⚠️ Warning: Line '{line}' does not contain '--' separator. Skipping.")

                task_query_options_list.append(turn_query_options)

            except Exception as e:
                print(f"\n❌ Error during inference or parsing for record {i + 1}, turn {j + 1}: {e}")
                task_query_options_list.append([])

        record['query_options_list'] = task_query_options_list

    # --- Step 4: Save the Augmented Dataset ---
    print(f"\nSaving processed data to: {output_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for data in interaction_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("✅ Processing complete. Output saved.")


if __name__ == "__main__":
    main()