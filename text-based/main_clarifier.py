# main_clarifier.py

# -*- coding: utf-8 -*-

"""
A script for multi-turn clarification dialogues, supporting interactive or evaluation modes.
The script interacts with a large language model through three core modules:
Re-Analyzer: Analyzes the conversation history to determine if the task is clear.
Question-Asker: If the task is ambiguous, generates a clarifying question.
Summarizer: If the task is clear, generates a final summary.
Supports interactive mode (manual input) and evaluation mode (LLM simulates user responses).
"""

import argparse
import json
import os
import re
import ssl
from typing import List, Dict, Any, Tuple, Optional

# Third-party library imports
import torch
from cprint import cprint
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Global Configurations ---

# Resolve SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# --- Prompt Definitions ---

REANALYZER_PROMPT_V5 = """You are a highly efficient task analysis agent. Your goal is to analyze a user's request and conversation history to determine the next step. Your output must be extremely concise and structured.

--- INSTRUCTIONS ---
1.  Review the original task and the entire conversation history.
2.  **Crucially, accept all user-provided information as fact, even if it doesn't match any suggested options.** Your role is to integrate the user's input, not validate it.
3.  **Handle "No Preference": If a user states they have no preference for a detail (e.g., 'no preference', 'any', 'I don't care'), treat that detail as KNOWN and RESOLVED. Do not ask for it again.**
4.  Determine if you have enough information to create a final, actionable summary.
5.  If the goal is still **VAGUE**:
    *   First, briefly list what you already know.
    *   Then, identify **all relevant** details that are still missing. Do NOT list details already provided or resolved. **Your goal is to be thorough; it is better to list a potentially useful detail than to miss an important one.**
    *   Assign a priority level to each missing detail:
        - **Lv3 (Critical):** The task is impossible without this.
        - **Lv2 (Important):** This detail significantly refines the task.
        - **Lv1 (Helpful):** This detail adds useful context or could lead to a better final result, but is not strictly necessary.
6.  If the goal is **CLEAR**, simply state that.

--- OUTPUT FORMAT (VAGUE) ---
[CONCLUSION]: VAGUE
[KNOWN_INFO]: Destination is [User's Answer], Budget is [User's Answer], Environment is no preference, [etc...]
[REASONING]: A brief, one-sentence explanation of what is still needed.
[MISSING_DETAILS]:
- Detail Name (Lv3): Suggested options...
- Another Detail Name (Lv2): Suggested options...
- Minor Detail Name (Lv1): Suggested options...

--- OUTPUT FORMAT (CLEAR) ---
[CONCLUSION]: CLEAR
[REASONING]: All critical and important details like destination, duration, and budget have been gathered.
"""

QUESTION_ASKER_PROMPT = """You are a friendly, concise assistant. Your task is to ask a single, open-ended question to clarify a specific topic.

--- INSTRUCTIONS ---
1.  You will be given a `[DETAIL_TO_CLARIFY]` which may include some examples.
2.  Formulate a brief, friendly question about it.
3.  **Frame the options as suggestions or examples, not a restrictive list.** For instance, use "like...", "for example...", or simply ask the question directly.
4.  Output ONLY the question itself. No thoughts, no extra text.

--- EXAMPLE ---
INPUT: [DETAIL_TO_CLARIFY]: Budget: $500, $1000, $2000+
YOUR OUTPUT: Got it. What's your approximate budget for this trip? For example, are we looking at under $500, around $1000, or more?
"""

FINAL_SUMMARY_PROMPT = """You are a summarization bot. You will be given a conversation history. Your job is to create a final, concise summary of the user's goal.

--- INSTRUCTIONS ---
1.  Review the entire conversation.
2.  Synthesize all key user preferences into a single, comprehensive sentence.
3.  Output ONLY the summary. No thoughts, no extra text.
"""

USER_SIMULATOR_PROMPT = """You are an automated user simulator for testing a conversational AI. Your goal is to provide a brief, realistic, and specific answer to the assistant's question, based on the conversation history.

--- INSTRUCTIONS ---
1.  Review the `[CONVERSATION_HISTORY]` and the `[ASSISTANT_QUESTION]`.
2.  Provide a **specific, concrete example** as an answer. For instance, if asked for a destination, say "Tokyo" or "a beach trip to Bali", not "a city" or "somewhere warm".
3.  Keep your answer very short and natural, like a real user would type (usually just a few words).
4.  Do NOT ask questions back. Do NOT repeat the assistant's question or options.
5.  Be plausible and consistent with the conversation so far.
6.  Output ONLY your simulated answer.

--- EXAMPLE ---
[CONVERSATION_HISTORY]:
User: Help me plan a vacation.
[ASSISTANT_QUESTION]:
Got it. Where would you like to travel? For example, a bustling city, a quiet beach, or maybe the mountains?
--- YOUR OUTPUT ---
A quiet beach sounds nice
"""

# --- Model Path Configurations ---
# Maps model names to their default paths. Can be overridden by the --model_path command-line argument.
# DEFAULT_MODEL_PATHS = {
#     "Qwen2.5-7B-Instruct-AWQ": "Qwen/Qwen2.5-7B-Instruct-AWQ",
#     "Qwen2.5-7B-Instruct": ".../Qwen2.5-7B-Instruct/",
#     "Qwen2.5-14B-Instruct-AWQ": ".../models--Qwen--Qwen2.5-14B-Instruct-AWQ/snapshots/539535859b135b0244c91f3e59816150c8056698/",
#     "Qwen2.5-32B-Instruct-AWQ": ".../models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c/",
#     "Qwen2.5-72B-Instruct-AWQ": ".../models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217/",
#     "Mistral-7B-Instruct-v0.2": ".../models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c/",
#     "Mistral-7B-Instruct-v0.3": ".../models--mistralai--Mistral-7B-Instruct-v0.3/",
#     "Llama-2-7b-chat-hf": ".../Llama-2-7b-chat-hf/",
#     "Mistral-Interact": "../Tell_Me_More/models/MI-mc/",
#     # ... Special model, path point to a code directory
# }


# --- Core Functional Functions ---

def init_text_model(args: argparse.Namespace) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initializes and loads the model and tokenizer based on the provided model path and arguments.

    Args:
        args (argparse.Namespace): Command-line arguments containing the model path and other configurations.

    Returns:
        A tuple containing the loaded model and tokenizer. (model, tokenizer)
         For API-based models (like GPT-4), may return (None, None).

    Raises:
        ValueError: If the model name is invalid or cannot be loaded.
    """
    model_path = args.model_name_or_path
    cprint.info(f"Initializing model from: '{model_path}'...")

    # --- Special Model Loading Logic ---
    # Use 'in' to check for keywords in the path for special handling
    if "GPT-4" in model_path:
        cprint.ok("Using GPT-4 (API-based). No local model to load.")
        return None, None

    if "Mistral-Interact" in model_path:
        # Special loading process for Mistral-Interact
        from model_center.tokenizer import LlamaTokenizer
        from model_center.model import Llama
        import bmtrain as bmt

        bmt.init_distributed(seed=0, zero_level=3)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = Llama.from_pretrained(model_path)
        cprint.ok("Mistral-Interact model loaded successfully.")
        return model, tokenizer

    # --- Standard Transformers Model Loading Logic ---
    device_map = "auto"
    torch_dtype = "auto"

    if "Mistral-7B-Instruct-v0.3" in model_path:
        torch_dtype = torch.bfloat16
    elif "Llama-2-7b-chat-hf" in model_path:
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="sdpa" if "Llama-2-7b-chat-hf" in model_path else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "Mistral-7B-Instruct-v0.2" in model_path:
        model.to("cuda")

    cprint.ok(f"Model from '{model_path}' and tokenizer loaded successfully.")
    return model, tokenizer


def inference_text_model(
        model: Any,
        tokenizer: Any,
        system_prompt: str,
        user_prompt: str,
        model_name_or_path: str,
        max_tokens: int = 512,
        do_sample: bool = False
) -> str:
    """
    Performs inference using the loaded model.

    Args:
        model: The loaded model object.
        tokenizer: The loaded tokenizer object.
        system_prompt: The system prompt.
        user_prompt: The user input/task.
        model_name_or_path: The model path or name, used to select the correct inference logic.
        max_tokens: The maximum number of tokens to generate.
        do_sample: Whether to use sampling.

    Returns:
        The text response generated by the model.
    """
    if "GPT-4" in model_name_or_path:
        from utils import gpt_chatcompletion
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return gpt_chatcompletion(messages)

    if "Mistral-Interact" in model_name_or_path:
        messages = f"<s>User: {system_prompt}\n\nHere is the task:\n{user_prompt}\nAgent: "
        # model is actually a LlamaRandomSampling instance here
        return model.generate([messages], max_length=32768)[0]

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    # --- Specific inference templates for different models ---
    if "Mistral-7B-Instruct-v0.2" in model_name_or_path:
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        generated_ids = model.generate(encodeds, max_new_tokens=max_tokens, do_sample=do_sample)
        decoded = tokenizer.batch_decode(generated_ids)[0]
        inst_end = "[/INST]"
        response_start = decoded.find(inst_end)
        response = decoded[response_start + len(inst_end):].strip() if response_start != -1 else decoded
        return response.replace("</s>", "").strip()

    elif "Llama-2-7b-chat-hf" in model_name_or_path:
        full_prompt = f"System prompt: {system_prompt}\n\nuser prompt\n{user_prompt}\nAgent: "
        input_ids = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        output = model.generate(**input_ids, max_new_tokens=max_tokens)
        return tokenizer.decode(output[0], skip_special_tokens=True).replace(full_prompt, "")

    else:  # Suitable for standard chat template models like Qwen, Mistral v0.3, etc.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generate_kwargs = {"max_new_tokens": max_tokens, "do_sample": do_sample}
        if do_sample:
            generate_kwargs.update({"temperature": 0.7, "top_p": 0.9})

        generated_ids = model.generate(**model_inputs, **generate_kwargs)
        generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# --- Helper Utility Functions ---

def parse_options_from_detail_string(detail_str: str) -> List[str]:
    """Parses options from a string in the format 'Detail: opt1, opt2'."""
    if ':' not in detail_str:
        return []
    options_part = detail_str.split(':', 1)[1]
    return [opt.strip() for opt in options_part.split(',') if opt.strip()]


def parse_details_from_analyzer(analysis_text: str) -> List[str]:
    """Parses the list of missing details from the Re-Analyzer's output and sorts them by priority."""
    details = []
    if "[MISSING_DETAILS]:" in analysis_text:
        details_section = analysis_text.split("[MISSING_DETAILS]:", 1)[1].strip()
        # Regex to match "- Detail Name (LvX): options..."
        matches = re.findall(r"-\s*(.*?)\s*\((Lv\d)\):?(.*)", details_section)
        for match in matches:
            detail_name, level, options = match[0].strip(), match[1], match[2].strip()
            full_detail_str = f"{detail_name}: {options}" if options else detail_name
            details.append({'level': level, 'detail': full_detail_str})
    # Sort in the order of Lv3, Lv2, Lv1
    details.sort(key=lambda x: x['level'], reverse=True)
    return [d['detail'] for d in details]


def initialize() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-turn clarification dialogue script")
    parser.add_argument("--data_dir", type=str, default="./test.jsonl",
                        help="Path to the input JSONL file containing tasks.")
    parser.add_argument('--start_from', type=int, default=0,
                        help="The line number of the task to start from in the input file.")
    parser.add_argument('--output_dir', type=str, default="./output", help="Directory to save the output results.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to a local model or a Hugging Face model ID. "
                             "Example (HF): 'Qwen/Qwen2.5-7B-Instruct-AWQ'.")
    parser.add_argument('--eval_mode', action='store_true',
                        help='Enable evaluation mode, where user input is automatically simulated by the LLM.')

    args = parser.parse_args()
    return args


def load_raw_dataset(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Loads the task dataset from a file."""
    if not os.path.exists(args.data_dir):
        cprint.warn(f"Data file {args.data_dir} does not exist, creating a sample file.")
        with open(args.data_dir, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"task": "Help me plan a vacation."}) + "\n")

    tasks = []
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append({"task": json.loads(line)["task"]})

    cprint.ok(f"Loaded {len(tasks)} tasks from {args.data_dir}.")
    return tasks[args.start_from:]


# --- 主逻辑 ---

def main():
    """Main execution function."""
    args = initialize()
    if args.eval_mode:
        cprint.warn("--- Evaluation Mode Enabled ---")
        cprint.warn("User input will be automatically simulated by the LLM.")

    # Initialize the model
    model, tokenizer = init_text_model(args)
    inference_handler = model

    # Special handling for the Mistral-Interact inference handler
    if "Mistral-Interact" in args.model_name_or_path:
        from model_center.generation.llama import LlamaRandomSampling
        inference_handler = LlamaRandomSampling(model, tokenizer)

    raw_tasks = load_raw_dataset(args)

    # Set up output file
    os.makedirs(args.output_dir, exist_ok=True)
    mode_prefix = 'eval' if args.eval_mode else 'interactive'
    safe_model_name = args.model_name_or_path.replace("/", "_").replace("\\", "_")
    output_filename = f'{mode_prefix}_output_{safe_model_name}.jsonl'
    output_filepath = os.path.join(args.output_dir, output_filename)

    if os.path.exists(output_filepath):
        os.remove(output_filepath)
        cprint.warn(f"Deleted old output file: {output_filepath}")

    for i, data in enumerate(raw_tasks):
        task = data["task"]
        task_index = args.start_from + i
        cprint.err(f"\n~~~~~~~~~~~~~~~~~ Starting new task {task_index} ~~~~~~~~~~~~~~~~~")
        cprint.warn(f"User Task: {task}")

        # Initialize conversation state
        conversation_history = ""
        actions = [{"role": "User", "thought": None, "content": task, "type": "response"}]
        initial_thought = None
        is_initial_vague = False

        # Statistics
        asked_details = set()
        asked_details_with_options_count = 0
        total_options_count = 0
        user_provided_facts_count = 0

        turn_count = 0
        max_turns = 16

        # --- Main Dialogue Loop ---
        while turn_count < max_turns:
            turn_count += 1
            cprint.ok(f"\n--- Turn {turn_count} ---")

            # Step 1: Analyze current state
            cprint.warn("Step 1: Analyzing current state...")
            analyzer_input = f"Original Task: {task}\n\nConversation History:\n{conversation_history if conversation_history else 'No conversation yet.'}"
            analysis_response = inference_text_model(
                inference_handler, tokenizer, REANALYZER_PROMPT_V5, analyzer_input, args.model_name_or_path
            )

            try:
                conclusion = re.search(r"\[CONCLUSION\]:\s*(\w+)", analysis_response).group(1)
                reasoning_match = re.search(r"\[REASONING\]:(.*?)(?=\[MISSING_DETAILS\]:|\[KNOWN_INFO\]:|$)",
                                            analysis_response, re.DOTALL)
                current_thought = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
                cprint.ok(f"Analysis Result: {conclusion} - {current_thought}")
            except AttributeError:
                cprint.fatal(f"Failed to parse Re-Analyzer output:\n{analysis_response}")
                break

            if turn_count == 1:
                initial_thought = current_thought
                is_initial_vague = (conclusion == "VAGUE")

            # Step 2a: If the task is vague, ask a clarifying question
            if conclusion == "VAGUE":
                missing_details_queue = parse_details_from_analyzer(analysis_response)
                if not missing_details_queue:
                    cprint.err("Analyzer concluded VAGUE but provided no details. Forcing summarization.")
                    conclusion = "CLEAR"  # Force entering the summarization process
                else:
                    detail_to_ask = missing_details_queue[0]
                    detail_name = detail_to_ask.split(':', 1)[0]
                    options = parse_options_from_detail_string(detail_to_ask)

                    # Update statistics
                    asked_details.add(detail_name)
                    if options:
                        asked_details_with_options_count += 1
                        total_options_count += len(options)

                    cprint.warn(f"Step 2: Decided to ask about '{detail_name}'")
                    question = inference_text_model(
                        inference_handler, tokenizer, QUESTION_ASKER_PROMPT, f"[DETAIL_TO_CLARIFY]: {detail_to_ask}",
                        args.model_name_or_path
                    )
                    cprint.info(f"Assistant: {question}")

                    actions.append(
                        {"role": "Assistant", "thought": current_thought, "content": question, "type": "response",
                         "option_num": len(options), "inappropriate_option_num": 0})

                    agent_response_for_history = f"Agent: {question}"
                    conversation_history += f"\n{agent_response_for_history}" if conversation_history else agent_response_for_history

                    user_input = ""
                    if args.eval_mode:
                        # Use LLM to simulate user response
                        simulator_prompt = f"[CONVERSATION_HISTORY]:\n{conversation_history}\n\n[ASSISTANT_QUESTION]:\n{question}"
                        user_input = inference_text_model(
                            inference_handler, tokenizer, USER_SIMULATOR_PROMPT, simulator_prompt, args.model_name_or_path,
                            do_sample=False)
                        cprint.info(f"Your Answer (LLM Simulated): {user_input}")
                    else:
                        user_input = input("Your Answer: ")
                        if not user_input.strip(): user_input = "(No response)"

                    # Tally user-provided information
                    non_info_responses = {"no preference", "any", "i don't care", "i see, please continue.",
                                          "(No response)"}
                    if user_input.lower().strip() not in non_info_responses:
                        user_provided_facts_count += 1

                    actions.append({"role": "User", "thought": None, "content": user_input, "type": "response"})
                    user_response_for_history = f"User: {user_input}"
                    conversation_history += f"\n{user_response_for_history}"

            # Step 2b: If the task is clear, generate the final summary
            if conclusion == "CLEAR":
                cprint.warn("Step 2: Decided to summarize.")
                summary_input = f"Original Task: {task}\n\nFull Conversation:\n{conversation_history}"
                final_summary = inference_text_model(
                    inference_handler, tokenizer, FINAL_SUMMARY_PROMPT, summary_input, args.model_name_or_path)
                cprint.ok(f"Final Summary: {final_summary}")
                actions.append(
                    {"role": "Assistant", "thought": current_thought, "content": final_summary, "type": "summary"})
                break

        # --- Loop ended, organizing and saving results ---
        user_record = {
            "missing_details_num": len(asked_details),
            "missing_with_options": asked_details_with_options_count,
            "total_options": total_options_count,
            "inappropriate_options": 0,
            "inappropriate_options_reason": None,
            "total_user_details": user_provided_facts_count,
            "user_details_in_summary": user_provided_facts_count,  # Assume all user-provided details are summarized
        }

        final_output = {
            "initial_thought": initial_thought,
            "user_record": user_record,
            "vague": is_initial_vague,
            "actions": actions,
        }

        cprint.ok(f"Saving results for task {task_index} to {output_filename}...")
        with open(output_filepath, "a", encoding='utf-8') as fout:
            fout.write(json.dumps(final_output, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    """
    # Use the Qwen2.5-7B model, with the path from the default dictionary, and enable evaluation mode
    python your_script_name.py --model_name Qwen2.5-7B-Instruct-AWQ --data_dir ./test.jsonl --eval_mode
    
    # Use the Llama-2-7b model and manually specify a new path with --model_path
    python your_script_name.py --model_name Llama-2-7b-chat-hf --model_path /path/to/my/llama_model --eval_mode
    
    # Use GPT-4 (no local model required)
    python your_script_name.py --model_name GPT-4 --data_dir ./test.jsonl --eval_mode
    
    # Run in interactive mode
    python your_script_name.py --model_name Qwen2.5-7B-Instruct-AWQ
    """
    main()

