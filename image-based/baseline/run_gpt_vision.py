import os
import argparse
import base64
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
# The model to be used for the vision task. 'gpt-4o' is recommended for its performance and cost-effectiveness.
GPT_MODEL = "gpt-4o"

# This system prompt is crucial. It instructs the AI on its role, the criteria for evaluation,
# and the exact format of the desired output.
SYSTEM_PROMPT = """You are an expert visual quality assessment assistant. Your task is to analyze an image provided by the user and determine if the specified object within the image is clear, complete, and well-centered.

Your evaluation must follow these criteria:
1.  **Completeness**: Is the object fully captured, or is a part of it cut off at the edge of the frame?
2.  **Clarity**: Is the image blurry or out of focus?
3.  **Distance**: Is the object too large (too close) or too small (too far) in the frame?
4.  **Composition**: Is the object reasonably centered in the image?

Based on your assessment, provide a single, concise, and actionable instruction in simple English. Your response should be the instruction itself, without any extra phrases like "Okay, based on my analysis...".

- If the object is perfectly displayed: "The image is clear, the '{object_name}' is fully in frame, and you can proceed."
- If the object is cut off: Indicate the direction to move the camera, e.g., "Please move the camera down and to the right to center the '{object_name}'."
- If it's blurry: "It's too blurry. Please adjust the focus or hold steady."
- If it's too close: "Please move further away. The '{object_name}' is too large in the frame and might be cut off."
- If it's too far: "Please move closer. The '{object_name}' is too small in the frame to see details."

Output only the final instruction.
"""


class GptVisionProcessor:
    """
    A class to process a dataset of images using the GPT-4 Vision API.
    It reads an Excel file, sends each image for evaluation, and saves the results.
    """

    def __init__(self, api_key: str, input_path: str, output_path: str, limit: int = None):
        """
        Initializes the processor.

        Args:
            api_key (str): The OpenAI API key.
            input_path (str): Path to the input Excel file.
            output_path (str): Path to save the output Excel file.
            limit (int, optional): The maximum number of rows to process. Defaults to None (all rows).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.data_limit = limit
        self.client = OpenAI(api_key=api_key)
        self.dataframe = None

    def run(self):
        """Executes the full processing pipeline."""
        if not self._load_and_prepare_data():
            return

        responses = self._process_rows()
        self.dataframe['gpt4_baseline_response'] = responses
        self._save_results()

    def _load_and_prepare_data(self) -> bool:
        """Loads the Excel file and applies the data limit if specified."""
        try:
            self.dataframe = pd.read_excel(self.input_path)
            print(f"Successfully read {len(self.dataframe)} records from: {self.input_path}")
        except FileNotFoundError:
            print(f"Error: Input Excel file not found at -> {self.input_path}")
            return False

        if self.data_limit is not None:
            self.dataframe = self.dataframe.head(self.data_limit)
            print(f"Note: Data has been limited to the first {self.data_limit} rows for processing.")

        return True

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str or None:
        """Encodes an image file to a Base64 string."""
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def _get_gpt_vision_response(self, user_prompt: str, base64_image: str) -> str:
        """
        Calls the GPT Vision API and handles retries.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ]
            }
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    temperature=0.2,  # Lower temperature for more deterministic output
                    max_tokens=150,
                    n=1,
                )
                content = response.choices[0].message.content
                return content.strip()
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if "content_policy_violation" in str(e):
                    return "API Error: Content policy violation."
                time.sleep(5)  # Wait before retrying

        print("Exceeded max retries. Skipping this image.")
        return "API Error: Failed after multiple retries."

    def _process_rows(self) -> list:
        """Iterates through DataFrame rows, processes each image, and collects responses."""
        responses = []
        for _, row in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0], desc="Processing Images"):
            image_path = row['image_path']
            # Use English object name if available, otherwise use a placeholder
            object_name = row.get('object_name_en', 'the object')

            user_prompt = f"Please assess the quality of the '{object_name}' in this image."

            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                responses.append("Error: Failed to read or encode image file.")
                continue

            response = self._get_gpt_vision_response(user_prompt, base64_image)
            # Inject the specific object name into the final response for clarity
            final_response = response.replace("{object_name}", object_name)
            responses.append(final_response)

        return responses

    def _save_results(self):
        """Saves the DataFrame with the new response column to an Excel file."""
        try:
            self.dataframe.to_excel(self.output_path, index=False, engine='openpyxl')
            print(f"\nProcessing complete! Results saved to: {self.output_path}")
        except Exception as e:
            print(f"\nError: Failed to save the Excel file. Reason: {e}")


def main():
    """
    Parses command-line arguments and initiates the vision processing task.
    """
    parser = argparse.ArgumentParser(
        description="Run GPT-4 Vision evaluation on a dataset of images specified in an Excel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # For better security, it's recommended to load the API key from an environment variable.
    # However, providing it as an argument is also supported.
    parser.add_argument(
        "-k", "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. It's recommended to set the OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "-i", "--input-excel",
        required=True,
        help="Path to the input Excel file containing image paths."
    )
    parser.add_argument(
        "-o", "--output-excel",
        required=True,
        help="Path to save the output Excel file with GPT-4V responses."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Limit the number of images to process (for testing purposes). Processes all if not set."
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: OpenAI API key is required. Please provide it via the --api-key argument or by setting the OPENAI_API_KEY environment variable.")
        return

    processor = GptVisionProcessor(
        api_key=args.api_key,
        input_path=args.input_excel,
        output_path=args.output_excel,
        limit=args.limit
    )
    processor.run()


if __name__ == '__main__':
    main()