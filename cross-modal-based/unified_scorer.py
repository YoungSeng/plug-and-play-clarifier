import argparse
import os
import re
import time
import pandas as pd
import torch
from tqdm import tqdm

# Constants for column names used in the input files.
# Modify these if your Excel files use different column headers.
# The original script used '口述问题' and 'gpt4o回答'.
QUESTION_COLUMN = "Question"
PREDICTION_COLUMN = "LLM_Answer"
CASE_ID_COLUMN = "Case_ID"


def initialize_gpt_client(api_key):
    """Initializes and returns the OpenAI client."""
    try:
        from openai import OpenAI
        print("🚀 Initializing OpenAI client...")
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully.")
        return client
    except ImportError:
        print("❌ 'openai' package not found. Please install it using: pip install openai")
        exit(1)


def initialize_qwen_model(model_path_or_name):
    """Loads and returns a local Hugging Face model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"🚀 Loading scoring model: {model_path_or_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        print(f"✅ Model loaded successfully to device: {model.device}.")
        return model, tokenizer
    except ImportError:
        print("❌ 'transformers' or 'torch' package not found. Please install them.")
        print("   pip install transformers torch accelerate bitsandbytes")
        exit(1)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Ensure the model path is correct and you have enough GPU memory.")
        exit(1)


def get_gpt_score(client, messages, model_name):
    """
    Sends a scoring request to the OpenAI API and returns the response content.
    Includes retry logic for handling API errors.
    """
    retries = 3
    for i in range(retries):
        try:
            print("📡 Requesting score from GPT...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ OpenAI API Error: {e}. Retrying in 5 seconds... ({i + 1}/{retries})")
            time.sleep(5)
    print(f"❌ Failed to get score from GPT after {retries} retries.")
    return f"Error: {e}"


def get_qwen_score(model, tokenizer, messages):
    """
    Generates a score using a local Qwen model.
    Includes retry logic for handling potential errors.
    """
    retries = 3
    for i in range(retries):
        try:
            print("📡 Requesting score from Qwen...")
            # Apply the chat template to format the conversation
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Generate a response with low temperature for deterministic output
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10,
                temperature=0.01,
                do_sample=False,
            )

            # Decode the generated tokens, skipping the input part
            input_ids_len = model_inputs.input_ids.shape[1]
            response_ids = generated_ids[0][input_ids_len:]
            content = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            return content
        except Exception as e:
            print(f"⚠️ Qwen model error: {e}. Retrying in 5 seconds... ({i + 1}/{retries})")
            time.sleep(5)
    print(f"❌ Failed to get score from Qwen after {retries} retries.")
    return f"Error: {e}"


def parse_score(score_text):
    """
    Parses the text response from the model to extract a float score between 0.0 and 1.0.
    """
    if not score_text:
        return None
    try:
        # Use regex to find a floating-point number (e.g., 1.0, 0.8)
        match = re.search(r"(\d\.\d+)", score_text)
        if match:
            score = float(match.group(1))
        else:
            # Fallback to direct conversion if regex fails
            score = float(score_text)

        if 0.0 <= score <= 1.0:
            return score
        else:
            print(f"⚠️ Score {score} is out of the valid range [0.0, 1.0].")
            return None
    except (ValueError, TypeError):
        print(f"⚠️ Could not parse score from model output: '{score_text}'")
        return None


def score_answer(scorer_type, question, ground_truth, prediction, model_components):
    """
    Constructs the prompt and calls the appropriate scoring function.

    Args:
        scorer_type (str): 'gpt' or 'qwen'.
        question (str): The question asked.
        ground_truth (str): The reference answer.
        prediction (str): The model's predicted answer.
        model_components (tuple): Contains client/model/tokenizer needed for scoring.

    Returns:
        float: The parsed score, or None if scoring fails.
    """
    system_prompt = """You are an assistant for scoring visual question answering tasks. Your goal is to determine if the predicted answer is semantically correct and consistent with the reference answer.

Focus only on semantic accuracy, not the phrasing. If the predicted answer is a full sentence, contains extra details, uses synonyms, or has a different subject, but the core meaning is correct, it should be considered "correct".

You will score each sample on a scale from 0.0 to 1.0 based on the following rules:
- Completely incorrect: 0.0
- Mostly incorrect: Below 0.3
- Partially correct, but with ambiguity or unclear semantics: 0.4–0.7
- Semantically consistent, with different phrasing: 0.8–1.0
- Completely equivalent or a reasonable, more detailed expression: 1.0

You must return only the numerical score.
"""

    few_shot_prompt = f"""
Example 1:
Question: What is the finger pointing at?
Reference Answer: a yellow bicycle
Predicted Answer: The finger is pointing at a yellow bicycle on the road.
Score: 1.0

Example 2:
Question: What is the finger pointing at?
Reference Answer: a trash can
Predicted Answer: It is pointing to a trash can, which seems to be for recycling.
Score: 1.0

Example 3:
Question: Which store is he pointing at?
Reference Answer: the Huawei store
Predicted Answer: The finger is pointing at the "Huawei" shop.
Score: 1.0

Now, please score the following sample:
Question: {question}
Reference Answer: {ground_truth}
Predicted Answer: {prediction}
Score (0.0 - 1.0):
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_prompt}
    ]

    if scorer_type == 'gpt':
        client, model_name = model_components
        score_text = get_gpt_score(client, messages, model_name)
    elif scorer_type == 'qwen':
        model, tokenizer = model_components
        score_text = get_qwen_score(model, tokenizer, messages)
    else:
        raise ValueError("Invalid scorer_type specified.")

    return parse_score(score_text)


def main(args):
    """
    Main function to orchestrate the scoring process.
    """
    # --- 1. Initialize Scoring Model ---
    model_components = None
    if args.scorer == 'gpt':
        if not args.api_key:
            raise ValueError("❌ OpenAI API key is required when using the 'gpt' scorer. "
                             "Provide it with --api_key or set the OPENAI_API_KEY environment variable.")
        client = initialize_gpt_client(args.api_key)
        model_components = (client, args.model_name_or_path)
    elif args.scorer == 'qwen':
        model, tokenizer = initialize_qwen_model(args.model_name_or_path)
        model_components = (model, tokenizer)

    # --- 2. Load Data ---
    # print("🔄 Loading data files...")
    # df_answers = pd.read_excel(args.gpt_answer_path)
    # df_standard = pd.read_excel(args.standard_path, header=None)
    #
    # # Extract standard answers. The original script assumes answers start
    # # from the 4th row (index 3) and are in the 15th column (index 14).
    # standard_answers = df_standard.iloc[3:, 14].reset_index(drop=True)

    print("🔄 Loading data files...")
    df_answers = pd.read_excel(args.gpt_answer_path)
    # The standard answers file has no header.
    df_standard = pd.read_excel(args.standard_path, header=None)

    # Extract standard answers from the third column (index 2).
    # The new format starts from the first row and the answer is in the 3rd column.
    standard_answers = df_standard.iloc[:, 2].reset_index(drop=True)

    if len(df_answers) != len(standard_answers):
        raise ValueError(
            "❌ The number of model answers does not match the number of standard answers. Please check the files.")

    # --- 3. Batch Scoring ---
    scores = []
    progress_bar_desc = f"🔍 Scoring with {args.scorer.upper()}"
    for i in tqdm(range(len(df_answers)), desc=progress_bar_desc):
        question = df_answers.iloc[i][QUESTION_COLUMN]
        prediction = df_answers.iloc[i][PREDICTION_COLUMN]
        ground_truth = standard_answers.iloc[i]
        case_id = df_answers.iloc[i].get(CASE_ID_COLUMN, i + 1)  # Fallback to index if column not found

        if pd.isna(prediction) or pd.isna(ground_truth):
            scores.append(None)
            continue

        score = score_answer(
            scorer_type=args.scorer,
            question=str(question),
            ground_truth=str(ground_truth),
            prediction=str(prediction),
            model_components=model_components
        )
        scores.append(score)
        print(f"\n--- Sample {case_id} ---\n"
              f"Q: {question}\n"
              f"GT: {ground_truth}\n"
              f"Pred: {prediction}\n"
              f"Score: {score}\n"
              f"---------------------")

    # --- 4. Save Results ---
    df_result = pd.DataFrame({
        "case_id": df_answers.get(CASE_ID_COLUMN, range(1, len(df_answers) + 1)),
        "question": df_answers[QUESTION_COLUMN],
        "ground_truth": standard_answers,
        "model_prediction": df_answers[PREDICTION_COLUMN],
        "score": scores,
    })

    df_result.to_excel(args.output_path, index=False)
    print(f"\n✅ Scoring complete. Results saved to: {args.output_path}")

    # --- 5. Calculate Average Score ---
    valid_scores = [s for s in scores if s is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"📊 Average score for the model: {avg_score:.4f}")
    else:
        print("⚠️ No valid scores were generated to calculate an average.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A unified script to score model answers using either GPT or a local Qwen model.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--gpt_answer_path",
        required=True,
        help="Path to the Excel file containing the model's answers to be scored."
    )
    parser.add_argument(
        "--standard_path",
        required=True,
        help="Path to the Excel file containing the ground-truth/standard answers."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the output Excel file with the scores."
    )
    parser.add_argument(
        "--scorer",
        type=str,
        required=True,
        choices=['gpt', 'qwen'],
        help="The scoring model to use: 'gpt' for OpenAI API or 'qwen' for a local Hugging Face model."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The identifier for the scoring model.\n"
             "For 'gpt', this is the model name (e.g., 'gpt-4o', 'gpt-4-turbo').\n"
             "For 'qwen', this is the path to the local model directory or its name on Hugging Face Hub."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="Your OpenAI API key. Required if --scorer is 'gpt'. Can also be set via the OPENAI_API_KEY environment variable."
    )

    args = parser.parse_args()
    main(args)