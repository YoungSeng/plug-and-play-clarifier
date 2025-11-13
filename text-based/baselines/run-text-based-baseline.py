# -*- coding: utf-8 -*-
"""
This script runs a text-based conversational baseline for evaluating an agent's ability
to understand and clarify a user's intent. It can operate in two modes:
1.  Interactive Mode: A human provides responses to the agent's questions.
2.  Evaluation Mode: A separate LLM (`USER_SIMULATOR`) automatically generates
    realistic user responses for hands-off evaluation.

This version has been modified to produce the exact same I/O and processing logic
as the reference `run_baselines.py` script, including its specific JSONL output format
and in-loop metrics calculation.
"""

import argparse
import json
import os
import re
import ssl
import time
import requests
from cprint import cprint

# --- Third-party LLM/Model Libraries ---
from openai import OpenAI
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Graceful Import for Optional Dependency ---
try:
    import bmtrain as bmt
except ImportError:
    bmt = None  # bmtrain is optional

# --- Global Configuration ---
ssl._create_default_https_context = ssl._create_unverified_context

# ==============================================================================
# 1. CONSTANT DEFINITIONS (PROMPTS)
# ==============================================================================

# NOTE: These prompts are preserved from the original scripts.

AGENT_PROMPT = """You are an agent trying to understand user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention. Here are steps in detail and corresponding EXAMPLEs.

--- Step 1: initial thought generation ---
1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.
2. List the important missing details and some according options if the task is vague.
### EXAMPLE:
[INITIAL THOUGHT] The task is to research methods to secure a home WiFi network. This is a broad topic...

--- Step 2: inquiry for more information if vague ---
1. If the task is vague, inquiry more details with options...
### EXAMPLE:
[INQUIRY THOUGHT] I need to understand which aspect of security...
[INQUIRY] Sure, I can help you with that! Are you looking for general best practices...

--- Step 3: summarize user's intention ---
1. Make the summary once information is enough...
### EXAMPLE:
[SUMMARY THOUGHT] The user has provided specific constraints...
[SUMMARY] The user seeks information on basic, general best practices..."""

USER_SIMULATOR_PROMPT = """You are an automated user simulator for testing a conversational AI. Your goal is to provide a brief, realistic, and specific answer to the assistant's question, based on the conversation history.
--- INSTRUCTIONS ---
1. Review the `[CONVERSATION_HISTORY]` and the `[ASSISTANT_QUESTION]`.
2. Provide a **specific, concrete example** as an answer.
3. Keep your answer very short and natural.
4. Do NOT ask questions back. Do NOT repeat the assistant's question or options.
5. Output ONLY your simulated answer.
--- EXAMPLE ---
[CONVERSATION_HISTORY]:
User: Help me plan a vacation.
[ASSISTANT_QUESTION]:
Got it. Where would you like to travel? For example, a bustling city, a quiet beach, or maybe the mountains?
--- YOUR OUTPUT ---
A quiet beach sounds nice
"""


# ==============================================================================
# 2. ARGUMENT PARSING
# ==============================================================================

def parse_arguments():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Run a text-based conversational baseline for agent evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("--agent_model_name_or_path", type=str, required=True,
                        help="Identifier for the agent model. Can be a HuggingFace ID, a local path, or an API model name (e.g., 'api/meta-llama/Llama-3.1-8B-Instruct').")
    parser.add_argument("--data_dir", type=str, default="./test.jsonl",
                        help="Path to the input data file containing user tasks.")
    parser.add_argument('--output_dir', type=str, default="./output",
                        help="Directory to save the output JSONL file.")

    # --- Evaluation Mode Arguments ---
    parser.add_argument('--eval_mode', action='store_true',
                        help='Enable evaluation mode, which uses an LLM to simulate user responses.')
    parser.add_argument("--simulator_model_name_or_path", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ",
                        help="Model identifier for the user simulator in evaluation mode.")

    # --- Other Optional Arguments ---
    parser.add_argument('--start_from', type=int, default=0,
                        help="The starting index of the task to process from the data file.")

    return parser.parse_args()


# ==============================================================================
# 3. MODEL INITIALIZATION
# ==============================================================================

def init_text_model(model_name_or_path, is_bmtrain_model=False):
    """
    Loads and initializes a text generation model and its tokenizer.
    Supports HuggingFace Hub IDs, local paths, and API model placeholders.
    """
    # --- Handle non-local models first ---
    if model_name_or_path.startswith("api/") or model_name_or_path == "GPT-4":
        cprint.info(f"Using API model: {model_name_or_path}. No local model will be loaded.")
        return None, None

    # --- Handle special bmtrain models ---
    if is_bmtrain_model:
        if bmt is None:
            raise ImportError(f"bmtrain is required for '{model_name_or_path}' but is not installed.")
        cprint.info(f"Loading bmtrain model from: {model_name_or_path}")
        from model_center.tokenizer import LlamaTokenizer
        from model_center.model import Llama
        bmt.init_distributed(seed=0, zero_level=3)
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = Llama.from_pretrained(model_name_or_path)
        return tokenizer, model

    # --- Standard HuggingFace Loader for all other cases ---
    cprint.info(f"Loading standard HuggingFace model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


# ==============================================================================
# 4. MODEL INFERENCE
# ==============================================================================

def inference_text_model(model, tokenizer, full_prompt, max_tokens=1024, do_sample=False, model_name=None):
    """
    Generates text using the provided model. This function routes the request to the
    correct inference logic based on the model_name. It expects a single, fully-formed
    prompt string.
    """
    # --- BMTrain-based Models ---
    if model_name in ["Mistral-Interact", "Mistral-7B-Instruct-v0.2"]:
        preds = model.generate([full_prompt], max_length=max_tokens, repetition_penalty=1.2, temperature=0.2,
                               top_p=0.95)
        return preds[0]

    # --- API-based Models ---
    elif model_name.startswith("api/"):
        cprint.ok(f"--- Calling API for model: {model_name} ---")
        api_model_name = model_name.replace("api/", "")

        # API messages format usually includes system and user roles.
        # Here, the full_prompt contains the user-side of the conversation.
        messages = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": full_prompt}
        ]

        # --- Active API Provider: Fireworks.ai ---
        # This implementation matches the one from the reference script.
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        # NOTE: The model in the payload is hardcoded as per the reference script.
        # For a more general solution, this could be mapped from `api_model_name`.
        payload = {
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
            "messages": messages
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_FIREWORKS_API_KEY"  # <-- IMPORTANT: Replace with your key
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_data = response.json()
            result = response_data["choices"][0]['message']["content"]
            return result.strip()
        except requests.exceptions.RequestException as e:
            cprint.fatal(f"API request failed: {e}")
            cprint.fatal(f"Response body: {response.text if 'response' in locals() else 'No response'}")
            return f"Error: API request failed. {e}"

    # --- Special Case: GPT-4 ---
    elif model_name == "GPT-4":
        from utils import gpt_chatcompletion  # Assumes a local utility file
        messages = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": full_prompt}
        ]
        return gpt_chatcompletion(messages)

    # --- Standard HuggingFace Models ---
    else:
        # For local models, we construct the chat from the system and user prompts.
        messages = [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": full_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generate_kwargs = {"max_new_tokens": max_tokens, "do_sample": do_sample}
        if do_sample:
            generate_kwargs.update({"temperature": 0.7, "top_p": 0.9})

        generated_ids = model.generate(**model_inputs, **generate_kwargs)
        # Decode only the newly generated tokens to avoid including the prompt
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ==============================================================================
# 5. DATA LOADING
# ==============================================================================

def load_raw_dataset(args):
    """
    Loads tasks from a JSONL file. Creates a default file if it doesn't exist.
    """
    tasks = []
    if not os.path.exists(args.data_dir):
        cprint.warn(f"Data file not found at {args.data_dir}. Creating a default file.")
        with open(args.data_dir, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"task": "Help me plan a vacation."}) + "\n")

    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append({"task": json.loads(line)["task"]})

    print(f"Loaded {len(tasks)} tasks. Starting from index {args.start_from}.")
    return tasks[args.start_from:]


# ==============================================================================
# 6. MAIN EXECUTION LOGIC
# ==============================================================================

def main():
    """
    Main function to run the conversational simulation and evaluation.
    """
    args = parse_arguments()
    if args.eval_mode:
        cprint.warn("--- EVALUATION MODE ENABLED ---")
        cprint.warn("User responses will be simulated by an LLM.")

    # --- Step 1: Load the main AGENT model ---
    cprint.ok(f"--- Loading Agent Model: {args.agent_model_name_or_path} ---")
    is_agent_bmtrain = 'MI-mc' in args.agent_model_name_or_path or 'Instruct-v0.2-mc' in args.agent_model_name_or_path or args.agent_model_name_or_path in [
        "Mistral-Interact", "Mistral-7B-Instruct-v0.2"]
    if is_agent_bmtrain:
        model_instance, tokenizer = init_text_model(args.agent_model_name_or_path, is_bmtrain_model=True)
        from model_center.generation.llama import LlamaRandomSampling
        model = LlamaRandomSampling(model_instance, tokenizer)
    else:
        model, tokenizer = init_text_model(args.agent_model_name_or_path)

    # --- Step 2: Load the USER SIMULATOR model if in eval mode ---
    simulator_model, simulator_tokenizer = None, None
    if args.eval_mode:
        cprint.warn(f"--- Loading User Simulator Model: {args.simulator_model_name_or_path} ---")
        if args.agent_model_name_or_path == args.simulator_model_name_or_path:
            cprint.info("Agent and Simulator models are the same. Re-using instance.")
            simulator_model, simulator_tokenizer = model, tokenizer
        else:
            simulator_model, simulator_tokenizer = init_text_model(args.simulator_model_name_or_path)
        if simulator_model is None and not args.simulator_model_name_or_path.startswith("api/"):
            cprint.fatal(f"Failed to load the user simulator model. Exiting.")
            return

    # --- Step 3: Load dataset and prepare output file ---
    raw_tasks = load_raw_dataset(args)
    os.makedirs(args.output_dir, exist_ok=True)
    mode_prefix = 'baseline_eval_output' if args.eval_mode else 'interactive_output'
    safe_model_name = args.agent_model_name_or_path.replace("/", "_")
    filename = f'{mode_prefix}_{safe_model_name}.jsonl'
    output_filepath = os.path.join(args.output_dir, filename)

    if os.path.exists(output_filepath):
        cprint.warn(f"Output file {output_filepath} already exists. It will be overwritten.")
        os.remove(output_filepath)

    # --- Step 4: Process each task in a conversational loop ---
    for i, data in enumerate(raw_tasks):
        task_index = args.start_from + i
        task = data["task"]
        cprint.err(f"\n~~~~~~~~~~~~~~~~~ Begin New Task {task_index} ~~~~~~~~~~~~~~~~~")
        cprint.warn(f"User Task: {task}")

        # Initialize conversation state (matches run_baselines.py)
        # The prompt is a single string, built turn-by-turn.
        # NOTE: The initial 'User:' part now contains the full AGENT_PROMPT, and the task is inside 'Here is the task:'.
        # This matches the prompt structure from `run_baselines.py`.
        prompt = f"Here is the task:\n{task}\nAgent: "

        # A separate history for the simulator's context and logging
        conversation_history_for_log = f"User: {task}"

        # Variables for storing results, as required by run_baselines.py output format
        actions = [{"role": "User", "thought": None, "content": task, "type": "response"}]
        initial_thought = ""
        is_initial_vague = False
        user_record = {}
        user_provided_facts_count = 0
        turn_count = 0
        MAX_TURNS = 16

        while turn_count < MAX_TURNS:
            turn_count += 1
            cprint.ok(f"\n--- Turn {turn_count} ---")
            cprint.warn("Step A: Agent is generating response...")

            # Generate the agent's raw response.
            # The full prompt now includes the AGENT_PROMPT via the inference function.
            # The `prompt` variable only contains the turn-by-turn dialog.
            agent_response_raw = inference_text_model(
                model, tokenizer, full_prompt=prompt, model_name=args.agent_model_name_or_path
            )

            # Post-processing for models that include the prompt in their output
            if agent_response_raw.startswith(prompt):
                agent_response_raw = agent_response_raw[len(prompt):]

            # Parse the agent's response based on [TAGS]
            is_summary = "[SUMMARY]" in agent_response_raw
            is_inquiry = "[INQUIRY]" in agent_response_raw
            thought, content = "", ""
            try:
                if is_summary:
                    thought = agent_response_raw.split("[SUMMARY THOUGHT]")[1].split("[SUMMARY]")[0].strip()
                    content = agent_response_raw.split("[SUMMARY]")[1].strip()
                    cprint.ok(f"Agent Thought: {thought}\nAgent Summary: {content}")
                elif is_inquiry:
                    thought = agent_response_raw.split("[INQUIRY THOUGHT]")[1].split("[INQUIRY]")[0].strip()
                    content = agent_response_raw.split("[INQUIRY]")[1].strip()
                    cprint.info(f"Agent Thought: {thought}\nAgent Question: {content}")
                else:
                    cprint.fatal(
                        f"Agent output is malformed (missing [INQUIRY] or [SUMMARY] tags):\n{agent_response_raw}")
                    break
            except IndexError as e:
                cprint.fatal(f"Failed to parse agent output (structure error: {e}):\n{agent_response_raw}")
                break

            # **LOGIC FROM run_baselines.py**: Parse initial thought on the first turn
            if turn_count == 1:
                if "[INITIAL THOUGHT]" in agent_response_raw:
                    initial_thought = agent_response_raw.split("[INITIAL THOUGHT]")[1].split("[INQUIRY THOUGHT]")[
                        0].strip()
                    is_initial_vague = not is_summary
                    missing_num = initial_thought.count("\n-")
                    missing_with_op = initial_thought.count(":")
                    options_list = re.findall(r':(.*?)(?:\n-|$)', initial_thought, re.DOTALL)
                    total_options = sum(
                        len(opt.strip().split(',')) for group in options_list for opt in group.split(',') if
                        opt.strip())
                    user_record = {"missing_details_num": missing_num, "missing_with_options": missing_with_op,
                                   "total_options": total_options, "inappropriate_options": 0,
                                   "inappropriate_options_reason": None}
                    cprint.ok(f"Auto-parsed Initial Thought Metrics: {user_record}")
                else:
                    initial_thought = "No [INITIAL THOUGHT] tag found."
                    is_initial_vague = False
                    user_record = {"missing_details_num": 0, "missing_with_options": 0, "total_options": 0,
                                   "inappropriate_options": 0, "inappropriate_options_reason": None}

            # Update prompt for the next turn
            prompt += agent_response_raw + "\n"

            if is_summary:
                actions.append({"role": "Assistant", "thought": thought, "content": content, "type": "summary"})
                cprint.ok("Conversation concluded with a summary.")
                break

            elif is_inquiry:
                # **LOGIC FROM run_baselines.py**: Log number of options offered
                option_num = len(content.split(',')) if '?' in content and ',' in content else 0
                actions.append({"role": "Assistant", "thought": thought, "content": content, "type": "response",
                                "option_num": option_num, "inappropriate_option_num": 0})

                conversation_history_for_log += f"\nAgent: {content}"
                user_input = ""

                # Get user response (simulated or stdin)
                if args.eval_mode:
                    cprint.warn(f"Step B: Simulator ({args.simulator_model_name_or_path}) is generating response...")
                    sim_prompt_context = f"[CONVERSATION_HISTORY]:\n{conversation_history_for_log}\n\n[ASSISTANT_QUESTION]:\n{content}"
                    sim_messages = [{"role": "system", "content": USER_SIMULATOR_PROMPT},
                                    {"role": "user", "content": sim_prompt_context}]
                    sim_text = simulator_tokenizer.apply_chat_template(sim_messages, tokenize=False,
                                                                       add_generation_prompt=True)
                    sim_inputs = simulator_tokenizer([sim_text], return_tensors="pt").to(simulator_model.device)
                    sim_ids = simulator_model.generate(sim_inputs.input_ids, max_new_tokens=200, do_sample=False)
                    user_input = simulator_tokenizer.batch_decode(sim_ids[:, sim_inputs.input_ids.shape[1]:],
                                                                  skip_special_tokens=True)[0].strip()
                    cprint.info(f"Simulated User Response: {user_input}")
                else:
                    user_input = input("Your Response: ")
                    if not user_input.strip():
                        user_input = "(No response provided)"

                user_provided_facts_count += 1
                actions.append({"role": "User", "thought": None, "content": user_input, "type": "response"})

                # Update prompt and history with user's response
                prompt += f"User: {user_input}\nAgent: "
                conversation_history_for_log += f"\nUser: {user_input}"

        if turn_count >= MAX_TURNS:
            cprint.fatal(f"Maximum turn limit of {MAX_TURNS} reached. Ending conversation.")

        # --- Step 5: Finalize and save result (matches run_baselines.py format) ---
        user_record["total_user_details"] = user_provided_facts_count
        user_record["user_details_in_summary"] = user_provided_facts_count  # This is a placeholder as in the original


        final_output = {
            "initial_thought": initial_thought,
            "user_record": user_record,
            "vague": is_initial_vague,
            "actions": actions,
        }

        cprint.ok(f"Saving results for task {task_index} to {output_filepath}...")
        with open(output_filepath, "a", encoding='utf-8') as fout:
            fout.write(json.dumps(final_output, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    """
    ----------------------------------------------------------------------
    HOW TO RUN THIS SCRIPT
    ----------------------------------------------------------------------

    The command-line arguments are now unified. Use `--agent_model_name_or_path` for all model types.

    1. For a Hugging Face Hub model in evaluation mode:
       (This will download the model to your cache)

       CUDA_VISIBLE_DEVICES=0 python your_script_name.py \\
           --agent_model_name_or_path "Qwen/Qwen2.5-7B-Instruct-AWQ" \\
           --simulator_model_name_or_path "Qwen/Qwen2.5-14B-Instruct-AWQ" \\
           --data_dir "./test.jsonl" \\
           --eval_mode

    2. For a local model in evaluation mode:

       CUDA_VISIBLE_DEVICES=0 python your_script_name.py \\
           --agent_model_name_or_path "/path/to/your/local/model" \\
           --simulator_model_name_or_path "/path/to/your/simulator/model" \\
           --data_dir "./test.jsonl" \\
           --eval_mode

    3. For an API-based model (e.g., Fireworks AI):
       (Remember to replace the placeholder API key in the script)

       python your_script_name.py \\
           --agent_model_name_or_path "api/meta-llama/Llama-3.1-8B-Instruct" \\
           --simulator_model_name_or_path "Qwen/Qwen2.5-14B-Instruct-AWQ" \\
           --data_dir "./test.jsonl" \\
           --eval_mode

    4. For interactive mode with a human user (no simulator needed):

       CUDA_VISIBLE_DEVICES=0 python your_script_name.py \\
           --agent_model_name_or_path "Qwen/Qwen2.5-7B-Instruct-AWQ" \\
           --data_dir "./test.jsonl"
    """
    main()