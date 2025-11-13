import argparse
import base64
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

try:
    from PIL import Image
except ImportError:
    print("❌ 'Pillow' package not found. Please install it using: pip install Pillow")
    exit(1)

# Constants for column names used in the input Excel file.
# The original script used '用例编号' (column 0) and '口述问题' (column 1).
CASE_ID_COLUMN_INDEX = 0
QUESTION_COLUMN_INDEX = 1


# --- Helper Function (previously in qwen_vl_utils.py) ---
def process_vision_info(messages):
    """
    Extracts image and video file paths from the message list.
    This helper is used by the Qwen model.
    """
    image_paths = []
    video_paths = []
    for msg in messages:
        if msg['role'] == 'user' and isinstance(msg['content'], list):
            for item in msg['content']:
                if item.get('type') == 'image':
                    image_paths.append(item['image'])
                elif item.get('type') == 'video':
                    video_paths.append(item['video'])
    return image_paths, video_paths


# --- Abstract Base Class for VQA Models ---
class VQAModel(ABC):
    """Abstract base class for a Visual Question Answering model."""

    def __init__(self, model_identifier):
        self.model_identifier = model_identifier

    @abstractmethod
    def get_answer(self, question: str, image_path: str) -> str:
        """
        Generates an answer for a given question and image path.
        Must be implemented by subclasses.
        """
        pass


# --- GPT-4o Implementation ---
class GPT_Model(VQAModel):
    """VQA model implementation using OpenAI's GPT API."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ 'openai' package not found. Please install it using: pip install openai")
            exit(1)

        print(f"🚀 Initializing OpenAI client for model: {model_name}...")
        self.client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully.")

    def _encode_image(self, image_path: str):
        """Encodes an image file to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"❌ Image not found at: {image_path}")
            return None

    def get_answer(self, question: str, image_path: str) -> str:
        base64_image = self._encode_image(image_path)
        if base64_image is None:
            return "Image not found"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }]

        retries = 3
        for i in range(retries):
            try:
                print("📡 Requesting answer from GPT API...")
                response = self.client.chat.completions.create(
                    model=self.model_identifier,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.2,  # Lower temperature for more deterministic answers
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ OpenAI API Error: {e}. Retrying... ({i + 1}/{retries})")
                time.sleep(5)

        return f"Error: Failed to get response after {retries} retries."


# --- Qwen-VL Implementation ---
class Qwen_Model(VQAModel):
    """VQA model implementation using a local Hugging Face Qwen-VL model."""

    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path)
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            print("❌ 'transformers' package not found. Please install it using: pip install transformers torch")
            exit(1)

        print(f"🚀 Loading Qwen-VL model from: {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        print(f"🔩 Using device: {self.device}")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto" if device == "auto" else self.device,
                attn_implementation="flash_attention_2",
            )
            print("✅ Flash Attention 2 enabled.")
        except (ImportError, RuntimeError):
            print("⚠️ Flash Attention 2 not available. Falling back to default implementation.")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto" if device == "auto" else self.device,
            )

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Qwen-VL model and processor loaded successfully.")

    def get_answer(self, question: str, image_path: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                # This part is just for constructing the message template.
                # The actual image loading happens next.
                {"type": "image", "image": image_path},
            ],
        }]

        try:
            print("📡 Generating answer with Qwen-VL...")
            text_template = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # --- START OF CHANGE ---
            # The original code passed the image path directly.
            # The processor expects loaded image objects.
            # We load the image from the path using Pillow here.
            image_paths, _ = process_vision_info(messages)
            loaded_images = [Image.open(p).convert("RGB") for p in image_paths]
            # --- END OF CHANGE ---

            inputs = self.processor(
                text=[text_template],
                # Pass the list of loaded PIL Image objects instead of paths
                images=loaded_images,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if 'assistant\n' in output_text:
                return output_text.split('assistant\n')[-1].strip()
            return output_text.strip()

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"


# --- Main Execution Logic ---
def main(args):
    # --- 1. Initialize the selected model ---
    if args.baseline == 'gpt':
        if not args.api_key:
            raise ValueError("❌ OpenAI API key is required for the 'gpt' baseline. "
                             "Provide it with --api_key or set the OPENAI_API_KEY environment variable.")
        model = GPT_Model(model_name=args.model_name_or_path, api_key=args.api_key)
    elif args.baseline == 'qwen':
        model = Qwen_Model(model_path=args.model_name_or_path, device=args.device)
    else:
        # This case should not be reached due to argparse 'choices'
        raise ValueError("Invalid baseline specified.")

    # --- 2. Load and prepare data ---
    # print(f"🔄 Loading data from: {args.data_path}")
    # df = pd.read_excel(args.data_path, header=None)
    # # Skip header rows and process the specified range
    # data_to_process = df.iloc[3:].iloc[args.start_index:args.end_index].reset_index(drop=True)
    # print(f"📊 Found {len(data_to_process)} samples to process.")

    # --- 2. Load and prepare data ---
    print(f"🔄 Loading data from: {args.data_path}")
    df = pd.read_excel(args.data_path, header=None)
    # The new format has no header rows to skip. Process the specified range directly.
    data_to_process = df.iloc[args.start_index:args.end_index].reset_index(drop=True)
    print(f"📊 Found {len(data_to_process)} samples to process.")

    # --- 3. Process each sample ---
    results = []
    for idx, row in tqdm(data_to_process.iterrows(), total=len(data_to_process),
                         desc=f"Processing with {args.baseline.upper()}"):
        case_id = str(row[CASE_ID_COLUMN_INDEX])
        question = str(row[QUESTION_COLUMN_INDEX])
        # Assume image format is .jpg, change if necessary
        image_path = os.path.join(args.image_dir, f"{case_id}.jpg")

        print(f"\n🔍 Processing sample [{args.start_index + idx}] | Case ID: {case_id}")

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            answer = "Image file is missing"
        else:
            start_time = time.time()
            answer = model.get_answer(question, image_path)
            end_time = time.time()
            print(f"✅ Answer: {answer}")
            print(f"⏱️ Time taken: {round(end_time - start_time, 2)} seconds")

        results.append([case_id, question, answer])

    # --- 4. Save results ---
    output_column_name = f"{args.baseline}_answer"
    df_out = pd.DataFrame(results, columns=["case_id", "Question", "LLM_Answer"])       # output_column_name

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_out.to_excel(args.output_path, index=False)

    print(f"\n📁 All results have been saved to: {args.output_path}")
    print(f"📊 Total samples processed: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VQA baselines (GPT-4o or Qwen-VL) on a dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument("--baseline", type=str, required=True, choices=['gpt', 'qwen'],
                        help="The VQA baseline model to use.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input Excel file containing questions and case IDs.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the directory containing the images.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output Excel file with the generated answers.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Identifier for the model.\n"
                             "For 'gpt': a model name like 'gpt-4o'.\n"
                             "For 'qwen': the path to the local model directory.")

    # --- Optional & Model-Specific Arguments ---
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key. Required if --baseline is 'gpt'. Can also be set via OPENAI_API_KEY env variable.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for the Qwen model (e.g., 'cuda:0', 'cpu'). 'auto' will use GPU if available. (Default: auto)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="The starting row index to process from the dataset (0-based, after skipping headers).")
    parser.add_argument("--end_index", type=int, default=10000,
                        help="The ending row index to process.")

    args = parser.parse_args()
    main(args)