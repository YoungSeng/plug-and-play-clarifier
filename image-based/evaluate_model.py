import os
import re
import argparse
import pandas as pd
from typing import Optional, Tuple

# --- Configuration: English Ground Truth Prompts ---
# These are the "perfect" prompts corresponding to each code.
# They are used to populate the 'ground_truth_prompt' column for easy comparison.
# Note: The '{object_name}' will be replaced by the object's English name.
PROMPT_TEMPLATES_EN = {
    0: "Please move the camera down and to the right to center the '{object_name}'.",
    # Corresponds to a top-left object
    1: "Please move the camera down to center the '{object_name}'.",  # Corresponds to a top object
    2: "Please move the camera down and to the left to center the '{object_name}'.",
    # Corresponds to a top-right object
    3: "Please move the camera to the left to center the '{object_name}'.",  # Corresponds to a right object
    4: "Please move the camera up and to the left to center the '{object_name}'.",
    # Corresponds to a bottom-right object
    5: "Please move the camera up to center the '{object_name}'.",  # Corresponds to a bottom object
    6: "Please move the camera up and to the right to center the '{object_name}'.",
    # Corresponds to a bottom-left object
    7: "Please move the camera to the right to center the '{object_name}'.",  # Corresponds to a left object
    8: "It's too blurry. Please adjust the focus or hold steady.",
    9: "Please move further away. The '{object_name}' is too large in the frame and might be cut off.",
    10: "Please move closer. The '{object_name}' is too small in the frame to see details.",
    11: "The image is clear and the '{object_name}' is fully in frame. You can proceed to the next step."
}


class ModelEvaluator:
    """
    A class to evaluate model responses for an image clarity/completeness task.
    It reads an Excel file with model predictions, classifies them, scores them
    against the ground truth, and generates a detailed report.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.dataframe = None
        self.response_column = None
        self.model_name = "Unknown Model"

    def run_evaluation(self):
        """Main method to run the entire evaluation pipeline."""
        if not self._load_data():
            return

        results = []
        for _, row in self.dataframe.iterrows():
            gt_code = self._extract_code_from_path(row['image_path'])
            pred_code = self._classify_response_to_code(row[self.response_column])
            score = self._calculate_score(gt_code, pred_code)
            gt_prompt = self._get_ground_truth_prompt(gt_code, row.get('object_name_en', 'object'))

            results.append({
                'ground_truth_code': gt_code,
                'predicted_code': pred_code,
                'score': score,
                'ground_truth_prompt': gt_prompt
            })

        # Add new columns to the original dataframe
        results_df = pd.DataFrame(results, index=self.dataframe.index)
        self.dataframe = pd.concat([self.dataframe, results_df], axis=1)

        self._print_summary_report()
        self._save_results()

    def _load_data(self) -> bool:
        """Loads the Excel file and identifies the model response column."""
        try:
            self.dataframe = pd.read_excel(self.input_path)
            print(f"Successfully read {len(self.dataframe)} records from: {self.input_path}")
        except FileNotFoundError:
            print(f"Error: Input Excel file not found at -> {self.input_path}")
            return False

        # Auto-detect response column and model name
        if 'model_feedback' in self.dataframe.columns:
            self.response_column = 'model_feedback'
            self.model_name = "'Ours' Model"
        elif 'gpt4_baseline_response' in self.dataframe.columns:
            self.response_column = 'gpt4_baseline_response'
            self.model_name = "GPT-4 Baseline"
        else:
            print("Error: Could not find a response column ('model_feedback' or 'gpt4_baseline_response').")
            return False

        print(f"Evaluating model: {self.model_name}")
        return True

    @staticmethod
    def _extract_code_from_path(image_path: str) -> int:
        """Extracts the numeric ground truth code from the image filename."""
        try:
            filename = os.path.basename(image_path)
            code_str = os.path.splitext(filename)[0]
            return int(code_str)
        except (ValueError, IndexError):
            return -1  # Return -1 for parsing errors

    @staticmethod
    def _classify_response_to_code(response: str) -> int:
        """Classifies the model's natural language response into a numeric code using English keywords."""
        if not isinstance(response, str):
            return -1  # Handle NaN or non-string inputs

        response_lower = response.lower()

        # --- Classification based on new English model outputs ---
        # The order of checks is important to avoid ambiguity.

        # Category 11 (OK)
        if re.search(r"clear|ready for the next step|fully visible", response_lower):
            return 11
        # Category 10 (Too Small / Far)
        if re.search(r"closer|too small", response_lower):
            return 10
        # Category 9 (Too Large / Close)
        if re.search(r"further away|too large", response_lower):
            return 9
        # Category 8 (Blurry)
        if re.search(r"blurry|unclear|hold steady|refocus", response_lower):
            return 8

        # Directional Categories (0-7)
        # Check for two-word directions first
        if "down" in response_lower and ("right" in response_lower or "to the right" in response_lower):
            return 0
        if "down" in response_lower and ("left" in response_lower or "to the left" in response_lower):
            return 2
        if "up" in response_lower and ("left" in response_lower or "to the left" in response_lower):
            return 4
        if "up" in response_lower and ("right" in response_lower or "to the right" in response_lower):
            return 6

        # Check for single-word directions
        if "down" in response_lower:
            return 1
        if "left" in response_lower:
            return 3
        if "up" in response_lower:
            return 5
        if "right" in response_lower:
            return 7

        # If no rules match, return -1 (unclassifiable)
        return -1

    @staticmethod
    def _calculate_score(gt_code: int, pred_code: int) -> float:
        """Calculates the score based on ground truth and predicted codes."""
        if gt_code == -1 or pred_code == -1:
            return 0.0  # Unclassifiable or parsing error gets 0 points

        if gt_code == pred_code:
            return 1.0  # Perfect match

        # Define "neighbor" relationships for directional codes to grant partial credit
        # Codes: 0:DL, 1:D, 2:DR, 3:R, 4:UR, 5:U, 6:UL, 7:L
        # Note: The mapping here seems to differ from the prompt comments. I will use a standard one.
        # Let's assume a clockwise mapping: 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW
        # Let's use YOUR original mapping:
        # 0:UL, 1:U, 2:UR, 3:R, 4:DR, 5:D, 6:DL, 7:L (from the old Chinese script)
        # But this doesn't match the new English prompts. Let's create a logical one:
        # The prompt for code 0 says "move down and right", so the object is Top-Left.
        neighbors = {
            0: [1, 7],  # Top-Left neighbors are Top (1) and Left (7)
            1: [0, 2],  # Top neighbors are Top-Left (0) and Top-Right (2)
            2: [1, 3],  # Top-Right neighbors are Top (1) and Right (3)
            3: [2, 4],  # Right neighbors are Top-Right (2) and Bottom-Right (4)
            4: [3, 5],  # Bottom-Right neighbors are Right (3) and Bottom (5)
            5: [4, 6],  # Bottom neighbors are Bottom-Right (4) and Bottom-Left (6)
            6: [5, 7],  # Bottom-Left neighbors are Bottom (5) and Left (7)
            7: [6, 0]  # Left neighbors are Bottom-Left (6) and Top-Left (0)
        }

        if gt_code in neighbors and pred_code in neighbors[gt_code]:
            return 0.5  # Partial credit for adjacent directions

        return 0.0  # Completely wrong

    @staticmethod
    def _get_ground_truth_prompt(code: int, object_name: str) -> str:
        """Generates the English ground truth prompt text from a code."""
        template = PROMPT_TEMPLATES_EN.get(code, "Invalid ground truth code.")
        return template.format(object_name=object_name)

    def _print_summary_report(self):
        """Calculates and prints the final evaluation summary."""
        print(f"\n--- {self.model_name} Evaluation Summary ---")
        total = len(self.dataframe)
        perfect = (self.dataframe['score'] == 1.0).sum()
        partial = (self.dataframe['score'] == 0.5).sum()
        wrong = (self.dataframe['score'] == 0.0).sum()

        strict_accuracy = perfect / total if total > 0 else 0
        loose_accuracy = (perfect + partial) / total if total > 0 else 0
        average_score = self.dataframe['score'].mean()

        print(f"Total Samples: {total}")
        print(f"Perfect Matches (1.0 pt): {perfect} ({strict_accuracy:.2%})")
        print(f"Partial Matches (0.5 pt): {partial} ({partial / total:.2%})")
        print(f"Incorrect Matches (0.0 pt): {wrong} ({wrong / total:.2%})")
        print("-" * 25)
        print(f"Strict Accuracy: {strict_accuracy:.2%}")
        print(f"Loose Accuracy: {loose_accuracy:.2%}")
        print(f"Average Score: {average_score:.4f}")
        print("-" * (len(self.model_name) + 22))

    def _save_results(self):
        """Saves the DataFrame with evaluation results to a new Excel file."""
        # Define the final column order for the output file
        final_columns = [
            'image_path',
            'object_name_en',
            'ground_truth_code',
            'predicted_code',
            'score',
            'ground_truth_prompt',
            self.response_column
        ]

        # Ensure all columns exist, then select them in the desired order
        output_df = self.dataframe[[c for c in final_columns if c in self.dataframe.columns]]

        try:
            output_df.to_excel(self.output_path, index=False, engine='openpyxl')
            print(f"\nEvaluation complete! Results saved to: {self.output_path}")
        except Exception as e:
            print(f"\nError: Failed to save the Excel file. Reason: {e}")


def main():
    """
    Main function to parse command-line arguments and run the evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate model responses from an Excel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input Excel file containing model responses."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the output Excel file with evaluation results."
    )
    args = parser.parse_args()

    evaluator = ModelEvaluator(input_path=args.input, output_path=args.output)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()