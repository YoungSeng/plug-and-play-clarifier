# -*- coding: utf-8 -*-
"""
This script evaluates the performance of a conversational model by comparing its
output against human-annotated references. It calculates several metrics,
including vagueness judgment accuracy and the recovery rate of missing details.

The script uses a specified Language Model (LLM) to perform semantic matching
between the model's generated questions and the human-provided intentions.

How to Run:
-----------
python your_script_name.py \
    --model_name_or_path "Qwen/Qwen3-8B-AWQ" \
    --input_split_file "./output/eval_output_Qwen2.5-7B-Instruct_split.jsonl" \
    --human_ref_file "./data/data_labeling/test_data_report_mix.jsonl" \
    --output_dir "./metrics"
"""

import os
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
        description="Evaluate conversational model performance against human references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to a local model or a Hugging Face model ID for semantic matching. "
             "Example (HF): 'Qwen/Qwen3-8B-AWQ'."
    )
    parser.add_argument(
        "--input_split_file",
        type=str,
        required=True,
        help="Path to the split JSONL file (output from the first script) containing model interactions."
    )
    parser.add_argument(
        "--human_ref_file",
        type=str,
        required=True,
        help="Path to the JSONL file containing human-annotated references/labels."
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./metrics",
        help="Directory where the final metrics JSON file will be saved."
    )

    return parser.parse_args()


# ==============================================================================
# 2. MODEL INITIALIZATION AND INFERENCE FUNCTIONS
# ==============================================================================

def init_qwen_model(model_path: str):
    """Initializes and loads the specified Hugging Face model and tokenizer."""
    print(f"Initializing model for semantic matching: {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✅ Model and tokenizer initialized successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        raise RuntimeError("Failed to initialize model for semantic matching.") from e


def qwen_semantic_match(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Uses the loaded Qwen model for a specific semantic matching task.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,  # Use deterministic generation
    )

    input_token_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_token_len:].tolist()

    try:
        # Some Qwen models use <think>...</think> tags. Strip this part.
        think_end_index = len(output_ids) - output_ids[::-1].index(151668)  # 151668 is '</think>'
        response_ids = output_ids[think_end_index:]
    except ValueError:
        response_ids = output_ids

    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response


# ==============================================================================
# 3. PROMPT DEFINITION FOR SEMANTIC MATCHING
# ==============================================================================

SYS_PROMPT = """You are a helpful assistant and good at judging similarities between phrases. Given a QUESTION and a list of phrases, you should determine if the provided QUESTION semantically matches any of the entries in the list. Directly answer with the phrase in the list if there is any semantic match, and 'None of the above' otherwise. Do not include any explanations and additional information. Remember to strictly follow the format given in the following examples:

1. Example with a semantic match:

### QUESTION:
What is the time frame for the price predictions?

### List of phrases:
- Historical data timeframe and granularity
- Criteria for efficiency
- Specific stocks or market sectors
- Computational resources available
- Type of historical data

### Answer:
Historical data timeframe and granularity

2. Example without a semantic match:

### QUESTION:
What is the time frame for the price predictions?

### List of phrases:
- Metrics or sources to determine popularity
- User's personal style preferences
- Geographic region
- Specific fashion categories

### Answer:
None of the above"""

USER_PROMPT_TEMPLATE = """Here is a QUESTION and a list of phrases:

### QUESTION:
{question}

### List of phrases:
{list_str}"""


# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================

def main():
    """Main function to run the evaluation pipeline."""
    args = get_args()

    # --- Print configuration for user confirmation ---
    print("--- Configuration ---")
    print(f"  Semantic Matching Model: {args.model_name_or_path}")
    print(f"  Input Split File:        {args.input_split_file}")
    print(f"  Human Reference File:    {args.human_ref_file}")
    print(f"  Output Directory:        {args.output_dir}")
    print("---------------------\n")

    # --- Step 1: Load Datasets ---
    for path in [args.input_split_file, args.human_ref_file]:
        if not os.path.exists(path):
            print(f"❌ Error: File not found at {path}")
            return

    print("Loading datasets...")
    interaction_dataset = [json.loads(line) for line in open(args.input_split_file, 'r', encoding='utf-8')]
    human_ref_dataset = [json.loads(line) for line in open(args.human_ref_file, 'r', encoding='utf-8')]
    print(
        f"Loaded {len(interaction_dataset)} interaction records and {len(human_ref_dataset)} human reference records.")

    # --- Step 2: Initialize Model and Results ---
    model, tokenizer = init_qwen_model(args.model_name_or_path)
    results = {}
    num_tasks = len(interaction_dataset)

    # --- Metric 1: Vagueness Judgement Accuracy ---
    align_cnt = sum(
        1 for i in range(num_tasks) if interaction_dataset[i].get('vague') == human_ref_dataset[i].get('user_vague'))
    vagueness_judgement_accuracy = align_cnt / num_tasks if num_tasks > 0 else 0
    results['vagueness_judgement_accuracy'] = vagueness_judgement_accuracy
    print(f"\nMetric 1: Vagueness Judgement Accuracy: {vagueness_judgement_accuracy:.4f}")

    # --- Metric 2: Average Conversation Rounds ---
    total_rounds = sum(len(rec['actions']) // 2 for rec in interaction_dataset)
    average_conversation_rounds = total_rounds / num_tasks if num_tasks > 0 else 0
    results['average_conversation_rounds'] = average_conversation_rounds
    print(f"Metric 2: Average Conversation Rounds: {average_conversation_rounds:.2f}")

    # --- Metric 3: Missing Details Recover Rate ---
    print("\nCalculating Metric 3: Missing Details Recover Rate...")
    missing_details_recover_rate = {
        # "1": {"rate": 0.0, "cnt": 0},
        # "2": {"rate": 0.0, "cnt": 0},
        "3": {"rate": 0.0, "cnt": 0},
        # "total_recover_rate": {"rate": 0.0, "cnt": 0},
    }

    for i in tqdm(range(num_tasks), desc="Evaluating Missing Details"):
        is_vague_model = interaction_dataset[i].get('vague', False)
        is_vague_human = human_ref_dataset[i].get('user_vague', False)

        if is_vague_model and is_vague_human:
            human_intentions = []
            human_intentions.extend(human_ref_dataset[i].get('user_approve', []))
            human_intentions.extend(human_ref_dataset[i].get('user_rectify', []))
            human_intentions.extend(human_ref_dataset[i].get('user_add', []))

            if not human_intentions:
                continue

            list_str = '\n'.join([f"- {info['description']}" for info in human_intentions])
            flag_dict = {info['description'].lower(): {"hit": False, "importance": info['importance']} for info in
                         human_intentions}

            for turn_info in interaction_dataset[i].get('query_options_list', []):
                for query_options in turn_info:
                    question = query_options.get('query', '')
                    if not question:
                        continue

                    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, list_str=list_str)
                    resp = qwen_semantic_match(model, tokenizer, SYS_PROMPT, user_prompt)

                    if resp != "None of the above":
                        resp_lower = resp.lower()
                        if resp_lower in flag_dict:
                            flag_dict[resp_lower]['hit'] = True
                        else:
                            print(f"\n⚠️ Warning on task {i}: Model response '{resp}' not in human intention list.")

            task_i_results = {}
            for k, v in flag_dict.items():
                imp_str = str(v['importance'])

                if imp_str != '3':
                    continue
                if imp_str not in task_i_results:
                    task_i_results[imp_str] = {'hit': 0, 'total': 0}
                task_i_results[imp_str]['total'] += 1
                if v['hit']:
                    task_i_results[imp_str]['hit'] += 1

            for imp, res in task_i_results.items():
                missing_details_recover_rate[imp]['rate'] += res['hit'] / res['total'] if res['total'] > 0 else 0
                missing_details_recover_rate[imp]['cnt'] += 1

            # total_hits = sum(1 for v in flag_dict.values() if v['hit'])
            # if len(flag_dict) > 0:
            #     missing_details_recover_rate['total_recover_rate']['rate'] += total_hits / len(flag_dict)
            #     missing_details_recover_rate['total_recover_rate']['cnt'] += 1

    # Finalize rates by averaging
    for key, val in missing_details_recover_rate.items():
        if val['cnt'] > 0:
            missing_details_recover_rate[key] = val['rate'] / val['cnt']
        else:
            missing_details_recover_rate[key] = 0.0

    results['missing_details_recover_rate'] = missing_details_recover_rate

    # --- Step 4: Save Results ---
    # Generate a descriptive output filename
    source_filename = os.path.splitext(os.path.basename(args.input_split_file))[0].replace('_split', '')
    analyzer_model_name = args.model_name_or_path.split('/')[-1]  # Get last part of path/name
    output_filename = f"metric_{source_filename}_analyzed_by_{analyzer_model_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    print("\n--- Final Results ---")
    print(json.dumps(results, indent=4))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Metrics successfully saved to: {output_path}")


if __name__ == "__main__":
    main()