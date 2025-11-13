import argparse
import json
import os
from cprint import cprint

# --- Constants ---

# The system prompt defines the agent's behavior.
SYSTEM_PROMPT = """You are an agent trying to understand user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention. Here are steps in detail and corresponding EXAMPLEs.

--- Step 1: initial thought generation ---
1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.
2. List the important missing details and some according options if the task is vague.
### EXAMPLE:
[INITIAL THOUGHT] The task is to research methods to secure a home WiFi network. This is a broad topic and could include a variety of methods such as changing default passwords, using WPA3 encryption, enabling a firewall, etc. However, the user has not specified any particular aspect of security they are interested in, the level of technical expertise they have, or if they are looking for basic or advanced security measures. These details would help narrow down the research to provide more targeted information. Some aspects of missiong details and potential options are as follows:
- Specific aspect of security: General best practices, Specific threats like hacking or malware, Parental controls and content filtering
- User's technical expertise: Beginner, Intermediate, Advanced
- Preference for security level: Basic security measures, Advanced security options

--- Step 2: inquiry for more information if vague ---
1. If the task is vague, inquiry more details with options according to the list in [INITIAL THOUGHT].
2. Think what information you have and what to inquiry next in [INQUIRY THOUGHT].
3. Present your inquiry with options for user to choose after [INQUIRY], and be friendly.
4. You could repeat Step 2 for multiple times (but less than 5 times), or directly skip Step 2 if user task is clear initially.
### EXAMPLE:
[INQUIRY THOUGHT] I need to understand which aspect of security the user wants to learn about. Starting with this will help tailor the information to their specific needs.
[INQUIRY] Sure, I can help you with that! Are you looking for general best practices to secure your WiFi network, or are you specifically concerned about certain threats like hacking or malware? Maybe you're interested in parental controls and content filtering?

--- Step 3: summarize user's intention ---
1. Make the summary once information is enough. You do not need to inquiry every missing details in [INITIAL THOUGHT].
2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be same as rounds of chatting.
3. Give final summary after [SUMMARY] with comprehensive details in one or two sentences.
### EXAMPLE:
[SUMMARY THOUGHT] The user has provided specific constraints over the course of three interactions which now allow for a clear understanding of their intention. Here are the user preferences and constraints:
- Focus on general best practices to secure WiFi network.
- User is technically advanced.
- Looking for basic security measures only.
[SUMMARY] The user seeks information on basic, general best practices for securing a home WiFi network suitable for someone with an advanced level of technical expertise."""


# --- Argument Parsing ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run interactive inference with a clarification agent.")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct-AWQ',
        choices=['Qwen/Qwen2.5-7B-Instruct-AWQ', 'Mistral-Interact'],
        help="The name of the model to use."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='path/to/your/local_model',  # Replaced absolute path with a placeholder
        help="Path to a local model, required for 'Mistral-Interact'."
    )

    # I/O arguments
    parser.add_argument('--start_from', type=int, default=0, help="Starting index for processing (if applicable).")
    parser.add_argument('--output_dir', type=str, default="./output", help="Directory to save output files.")

    return parser.parse_args()


# --- Model Initialization ---

def init_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct-AWQ"):
    """Initializes a Qwen model and tokenizer from Hugging Face."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cprint.info(f"Initializing Qwen model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cprint.ok("Qwen model initialized successfully.")
    return model, tokenizer


def init_mistral_model_and_generator(model_path):
    """
    Initializes a local 'Mistral-Interact' model, tokenizer, and generator.
    Note: Requires specific libraries like 'model_center' and 'bmtrain'.
    """
    # Check if the placeholder path is still being used
    if model_path == 'path/to/your/local_model':
        raise ValueError("Please provide a valid path to your local model using --model_name_or_path")

    from model_center.tokenizer import LlamaTokenizer
    from model_center.model import Llama
    from model_center.generation.llama import LlamaRandomSampling
    import bmtrain as bmt

    cprint.info(f"Initializing Mistral-Interact model from: {model_path}...")

    # Setup bmtrain
    bmt.init_distributed(seed=0, zero_level=3)

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load model
    model = Llama.from_pretrained(model_path)

    # Setup generator
    generator = LlamaRandomSampling(model, tokenizer)
    cprint.ok("Mistral-Interact model initialized successfully.")

    return model, tokenizer, generator


# --- Inference Logic ---

def build_prompt(task, model_name):
    """Builds the initial prompt based on the model type."""
    if model_name.startswith("Qwen"):
        return f"User: Here is the task:\n{task}\nAgent: "
    elif model_name == "Mistral-Interact":
        return f"<s>User: {SYSTEM_PROMPT}\n\nHere is the task:\n{task}\nAgent: "
    else:
        raise ValueError(f"Unsupported model for prompt building: {model_name}")


def run_inference(prompt, model, tokenizer, args, mistral_generator=None):
    """Runs a single inference pass and returns the model's prediction."""
    if args.model_name.startswith("Qwen"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.split("Agent: ")[0]}  # Extract user part
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            top_k=1,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    elif args.model_name == "Mistral-Interact":
        if mistral_generator is None:
            raise ValueError("Mistral generator is required for inference.")
        preds = mistral_generator.generate(
            [prompt],
            max_length=1024,
            repetition_penalty=1.2,
            temperature=0.2,
            top_p=0.95,
        )
        return preds[0]

    else:
        raise ValueError(f"Unsupported model for inference: {args.model_name}")


# --- Main Interaction Loop ---

def start_interactive_session(initial_task, args):
    """
    Manages the interactive conversation with the model to clarify the user's task.
    """
    # --- Step 1: Initialize Model ---
    model, tokenizer, mistral_generator = None, None, None
    if args.model_name.startswith("Qwen"):
        model, tokenizer = init_qwen_model(args.model_name)
    elif args.model_name == "Mistral-Interact":
        model, tokenizer, mistral_generator = init_mistral_model_and_generator(args.model_name_or_path)

    # --- Step 2: Prepare for Interaction ---
    task = initial_task
    prompt = build_prompt(task, args.model_name)

    actions = [{"role": "User", "thought": None, "content": task, "type": "response"}]
    save_dict = {}

    cprint.err("\n~~~~~~~~~~~~~~~~~ Begin New Task ~~~~~~~~~~~~~~~~~")
    cprint.warn(f"User Task: {task}")

    summary_flag = False
    is_task_vague = True

    # --- Step 3: Interaction Loop ---
    while not summary_flag:
        print("=-=-=-=-= Generating... =-=-=-=-=")
        pred = None
        for attempt in range(3):  # Retry up to 3 times on parsing errors
            try:
                pred = run_inference(prompt, model, tokenizer, args, mistral_generator)

                # Parse Initial Thought (only on the first turn)
                if "[INITIAL THOUGHT]" in pred and "initial_thought" not in save_dict:
                    initial_thought = pred.split("[INITIAL THOUGHT]")[1].split("[INQUIRY THOUGHT]")[0].strip()
                    save_dict["initial_thought"] = initial_thought
                    if "[SUMMARY]" in pred:
                        is_task_vague = False

                # Parse Summary or Inquiry
                if "[SUMMARY]" in pred:
                    thought = pred.split("[SUMMARY THOUGHT]")[1].split("[SUMMARY]")[0].strip()
                    response = pred.split("[SUMMARY]")[1].strip()
                    summary_flag = True
                else:
                    thought = pred.split("[INQUIRY THOUGHT]")[1].split("[INQUIRY]")[0].strip()
                    response = pred.split("[INQUIRY]")[1].strip()

                break  # Exit retry loop if parsing is successful

            except Exception as e:
                cprint.fatal(f"Parsing Error: {e}\nRaw Prediction: {pred}\nRetrying (attempt {attempt + 1}/3)...")

        if pred is None:
            cprint.fatal("Failed to get a valid response from the model after 3 attempts.")
            return "Interaction failed."

        # --- Step 4: Process and Display Response ---
        if summary_flag:
            cprint.ok(f"\nAssistant Final Summary: {response}")
            actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "summary"})
            save_dict["vague"] = is_task_vague
            # Final result is ready, exit loop
            return response
        else:
            cprint.info(f"\nAssistant Response: {response}")
            actions.append({"role": "Assistant", "thought": thought, "content": response, "type": "response"})
            prompt += pred + "\n"

        # --- Step 5: Get User Input and Continue ---
        user_input = input("Your Response: ")
        prompt += "User: " + user_input + "\nAgent: "
        actions.append({"role": "User", "thought": None, "content": user_input, "type": "response"})

    # --- Optional: Save test results ---
    # save_dict["prompt"] = prompt
    # save_dict["actions"] = actions
    # save_path = args.output_dir
    # os.makedirs(save_path, exist_ok=True)
    # print(f"Saving results to {save_path}...")
    # with open(os.path.join(save_path, f'user_interaction_record.jsonl'), "a", encoding='utf-8') as fout:
    #     fout.write(json.dumps(save_dict) + "\n")


if __name__ == '__main__':
    """
    Example Usage:

    # For Qwen (uses Hugging Face Hub, no local path needed)
    python your_script_name.py --model_name "Qwen/Qwen2.5-7B-Instruct-AWQ"

    # For Mistral-Interact (requires a local model path and specific libraries)
    # NOTE: You need to install 'bmtrain' and 'model_center' for this to work.
    # CUDA_VISIBLE_DEVICES=0 python your_script_name.py --model_name "Mistral-Interact" --model_name_or_path "/path/to/your/mistral/model"
    """

    args = parse_arguments()

    # You can start the session with a predefined task
    initial_task = "Help me arrange a trip to China."

    # Or take it from user input
    # initial_task = input("Give Your New Task: ")

    final_summary = start_interactive_session(initial_task, args)

    cprint.green("\n--- Interaction Complete ---")
    cprint.green(f"Final Summary: {final_summary}")