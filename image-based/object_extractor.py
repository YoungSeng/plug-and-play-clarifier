import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

# The system prompt defines the task for the language model.
SYSTEM_PROMPT = """
You are an intelligent assistant. Your task is to analyze the user's query and identify the core physical object the user wants the camera to focus on.
You must follow these rules for your response:
1. Return only the single most critical, visually detectable object name.
2. Use lowercase English words or phrases.
3. Do not include any explanations, punctuation, or other extraneous text.

Examples:
User query: "Help me see what's happening on this chessboard."
Your response: board
User query: "What does this QR code say?"
Your response: phone
User query: "What is this PowerPoint slide about?"
Your response: powerpoint
User query: "What does the instruction manual for this medicine say?"
Your response: instruction manual
"""

class ObjectExtractor:
    """
    A class to extract a key object from a text prompt using a causal language model.
    """
    def __init__(self, model_name_or_path: str):
        """
        Initializes the model and tokenizer.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model
                                      from the Hugging Face Hub.
        """
        print(f"Loading model: {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print("Model loaded successfully.")

    def extract_object(self, user_prompt: str) -> str:
        """
        Extracts the key object from a given user prompt.

        Args:
            user_prompt (str): The user's input text.

        Returns:
            str: The extracted object name in lowercase English.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Apply the chat template specific to the model
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate the response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=50,  # A short response is expected
            do_sample=False,
            temperature=0.0,
            top_k=1,
        )

        # Decode the generated tokens, skipping the prompt part
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()

def main():
    """
    Main function to run the script from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Extract a key object from a text prompt using an LLM."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="The model name or path from Hugging Face Hub."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The input prompt from the user."
    )
    args = parser.parse_args()

    try:
        extractor = ObjectExtractor(model_name_or_path=args.model_name_or_path)
        object_name = extractor.extract_object(user_prompt=args.prompt)
        print(f"\nInput Prompt: '{args.prompt}'")
        print(f"Extracted Object: '{object_name}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()