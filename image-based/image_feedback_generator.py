# image_feedback_generator.py

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import traceback

# --- Optional Imports for Detection Models ---
try:
    from groundingdino.util.inference import load_model as dino_load_model, \
        load_image as dino_load_image, \
        predict as dino_predict, \
        annotate as dino_annotate
except ImportError:
    print("Warning: GroundingDINO library not found. The 'GroundingDINO' model will not be available.")
    print("Please follow the official guide to install: 'pip install --no-build-isolation -e GroundingDINO'")
    dino_load_model = None  # To prevent runtime errors if not installed

try:
    from ultralytics import YOLOE
except ImportError:
    print("Warning: ultralytics library not found. The 'YOLOE' model will not be available.")
    YOLOE = None  # To prevent runtime errors if not installed

# --- Script Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Analysis Hyperparameters ---

# Blurriness Detection
LAPLACIAN_THRESHOLD = 50.0
FFT_BLUR_THRESHOLD = 0.45
FFT_RADIUS_RATIO = 0.1
LAPLACIAN_NORM_MIN = 10.0
LAPLACIAN_NORM_MAX = 300.0
FFT_NORM_MIN = 0.35
FFT_NORM_MAX = 0.60
WEIGHT_LAPLACIAN = 0.5
WEIGHT_FFT = 0.5
COMBINED_BLUR_THRESHOLD = 0.5

# Object Feedback Logic
EDGE_TOLERANCE = 5
TOO_SMALL_THRESH = 0.1
TOO_LARGE_THRESH = 0.9
LARGE_OBJECT_CHECK_METHOD = "AREA"  # 'AREA' or 'DIMENSION'
CENTER_TOLERANCE_X = 0.99
CENTER_TOLERANCE_Y = 0.99


# --- Clarity Analysis Functions ---

def normalize_score(score: float, min_val: float, max_val: float, invert: bool = True) -> float:
    """Normalizes a score to a 0-1 range."""
    if score >= max_val:
        normalized = 1.0
    elif score <= min_val:
        normalized = 0.0
    else:
        normalized = (score - min_val) / (max_val - min_val)
    return 1.0 - normalized if invert else normalized


def check_image_clarity_combined(image: np.ndarray) -> (bool, float, str):
    """
    Combines Laplacian and FFT methods to determine if an image is blurry.
    Returns: (is_blurry, final_blur_score, details_string)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Laplacian Variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # FFT
    h, w = gray.shape
    if h == 0 or w == 0:
        return False, 0.0, "FFT: Invalid ROI"

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    crow, ccol = h // 2, w // 2
    radius = int(min(h, w) * FFT_RADIUS_RATIO)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    total_energy = np.sum(np.abs(fshift))
    high_freq_energy = np.sum(np.abs(fshift[mask == 0]))
    fft_score = high_freq_energy / total_energy if total_energy > 0 else 0

    details_str = f"Laplace: {lap_var:.2f}, FFT: {fft_score:.4f}"

    # Weighted Score Combination
    lap_blur_degree = normalize_score(lap_var, LAPLACIAN_NORM_MIN, LAPLACIAN_NORM_MAX, invert=True)
    fft_blur_degree = normalize_score(fft_score, FFT_NORM_MIN, FFT_NORM_MAX, invert=True)
    final_blur_score = (WEIGHT_LAPLACIAN * lap_blur_degree) + (WEIGHT_FFT * fft_blur_degree)

    is_blur = final_blur_score > COMBINED_BLUR_THRESHOLD
    return is_blur, final_blur_score, details_str


# --- Core Feedback Generation Function ---

def get_feedback_for_object(box_xyxy: np.ndarray, image: np.ndarray, object_name: str) -> (str, bool, float):
    """
    Analyzes an object's bounding box and ROI to generate user feedback.

    Returns: (feedback_message, is_blurry, clarity_score)
    """
    img_h, img_w, _ = image.shape
    x1, y1, x2, y2 = box_xyxy

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return (f"Detected object '{object_name}' has invalid dimensions.", None, None)

    box_area = box_w * box_h
    img_area = img_w * img_h
    area_ratio = box_area / img_area

    # 1. Check if object is too large
    is_too_large = False
    if LARGE_OBJECT_CHECK_METHOD.upper() == 'DIMENSION':
        if (box_w / img_w) > TOO_LARGE_THRESH or (box_h / img_h) > TOO_LARGE_THRESH:
            is_too_large = True
    else:  # Default to 'AREA'
        if area_ratio > TOO_LARGE_THRESH:
            is_too_large = True

    if is_too_large:
        return (
        f"Please move further away. The '{object_name}' is too large in the frame and may be incomplete.", None, None)

    # 2. Check if object touches edges (cropped)
    is_touching_left = x1 <= EDGE_TOLERANCE
    is_touching_right = x2 >= img_w - EDGE_TOLERANCE
    is_touching_top = y1 <= EDGE_TOLERANCE
    is_touching_bottom = y2 >= img_h - EDGE_TOLERANCE

    edge_move_x = "left" if is_touching_left else "right" if is_touching_right else ""
    edge_move_y = "up" if is_touching_top else "down" if is_touching_bottom else ""
    if edge_move_x or edge_move_y:
        # Note: Direction is where the camera should point
        direction = " ".join(filter(None, [edge_move_y, edge_move_x]))
        return (
        f"Part of the object might be off-screen. Please move the camera {direction} to show the full '{object_name}'.",
        None, None)

    # 3. Check if object is centered
    box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left_bound, right_bound = img_w * (0.5 - CENTER_TOLERANCE_X / 2), img_w * (0.5 + CENTER_TOLERANCE_X / 2)
    top_bound, bottom_bound = img_h * (0.5 - CENTER_TOLERANCE_Y / 2), img_h * (0.5 + CENTER_TOLERANCE_Y / 2)

    center_move_x = "left" if box_center_x > right_bound else "right" if box_center_x < left_bound else ""
    center_move_y = "up" if box_center_y > bottom_bound else "down" if box_center_y < top_bound else ""
    if center_move_x or center_move_y:
        direction = " ".join(filter(None, [center_move_x, center_move_y]))
        return (f"Please move the camera {direction} to center the '{object_name}'.", None, None)

    # 4. Check if object is too small
    if area_ratio < TOO_SMALL_THRESH:
        return (f"Please move closer. The '{object_name}' is too small to see details.", None, None)

    # 5. Final Check: Clarity (Blur)
    roi = image[int(y1):int(y2), int(x1):int(x2)]
    if roi.shape[0] < 10 or roi.shape[1] < 10:
        return (f"Please move closer. The '{object_name}' is too small to check for clarity.", None, None)

    is_blur, clarity_score, details = check_image_clarity_combined(roi)
    if is_blur:
        return (
        f"The '{object_name}' is too blurry (score: {clarity_score:.4f}). Please hold steady and refocus. ({details})",
        True, clarity_score)
    else:
        return (
        f"Image is clear (score: {clarity_score:.4f}). The '{object_name}' is fully visible and centered. Ready for the next step.",
        False, clarity_score)


# --- Model and Processing Functions ---

def load_detection_model(args):
    """Loads the specified object detection model based on arguments."""
    model_type = args.model_type.upper()
    print(f"Loading {model_type} model from: {args.model_path}")

    if model_type == 'GROUNDINGDINO':
        if dino_load_model is None:
            raise ImportError("GroundingDINO library is not installed.")
        if not args.config_path:
            raise ValueError("A --config_path is required for GroundingDINO.")
        model = dino_load_model(args.config_path, args.model_path, device=DEVICE)
        if args.fp16 and DEVICE == 'cuda':
            model.half()
        return model

    elif model_type == 'YOLOE':
        if YOLOE is None:
            raise ImportError("Ultralytics library is not installed.")
        return YOLOE(args.model_path)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}. Choose 'GroundingDINO' or 'YOLOE'.")


def process_dataset(args):
    """Main function to run inference over the dataset."""
    # 1. Load Model
    try:
        model = load_detection_model(args)
        print(f"{args.model_type} model loaded successfully on {DEVICE}.")
    except (ImportError, ValueError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        return

    # 2. Prepare I/O
    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)
        print(f"Visualization images will be saved to: {args.viz_dir}")

    try:
        df = pd.read_excel(args.input_excel)
    except FileNotFoundError:
        print(f"Error: Input Excel file not found at {args.input_excel}")
        return

    if args.limit:
        df = df.head(args.limit)
        print(f"Warning: Processing limited to the first {args.limit} rows.")

    # 3. Process each row
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing with {args.model_type}"):
        try:
            image_path = row['image_path']
            # Use English name for model prompt, Chinese name for user feedback
            object_name_en = row['object_name_en']
            object_name_zh = row['object_name_zh']

            if not os.path.exists(image_path):
                results.append("Image file not found.")
                continue

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                results.append("Failed to read image file.")
                continue

            # --- Detection Logic ---
            box_xyxy = None
            img_with_box = img_bgr.copy()

            if args.model_type.upper() == 'GROUNDINGDINO':
                img_src, img_tensor = dino_load_image(image_path)
                if args.fp16 and DEVICE == 'cuda':
                    img_tensor = img_tensor.half()

                prompt = object_name_en.replace('_Ego4D', '').strip()
                boxes, logits, phrases = dino_predict(
                    model=model, image=img_tensor, caption=prompt,
                    box_threshold=args.box_thresh, text_threshold=args.text_thresh, device=DEVICE
                )
                if len(boxes) > 0:
                    best_idx = logits.argmax()
                    box_norm = boxes[best_idx]
                    H, W, _ = img_src.shape
                    box_tensor = box_norm * torch.Tensor([W, H, W, H]).to(box_norm.device)
                    box_tensor[:2] -= box_tensor[2:] / 2
                    box_tensor[2:] += box_tensor[:2]
                    box_xyxy = box_tensor.cpu().numpy()
                    img_with_box = dino_annotate(image_source=img_src, boxes=boxes, logits=logits, phrases=phrases)

            elif args.model_type.upper() == 'YOLOE':
                # YOLOE processing logic would go here
                # preds = model(img_bgr)
                # ... find best box for the target class ...
                # For now, this part is a placeholder
                print("YOLOE inference logic is not implemented in this version.")
                box_xyxy = None  # Placeholder

            # --- Feedback Generation ---
            if box_xyxy is not None:
                feedback, is_blur, score = get_feedback_for_object(box_xyxy, img_bgr, object_name_en)       # object_name_zh
                if args.visualize and is_blur is not None:
                    viz_text = f"Blur: {is_blur} (Score: {score:.4f})"
                    cv2.putText(img_with_box, viz_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                feedback = f"Object '{object_name_en}' not detected in the frame."      # object_name_zh

            results.append(feedback)

            # --- Save Visualization ---
            if args.visualize:
                safe_name = object_name_en.replace(' ', '_')
                img_basename = os.path.splitext(os.path.basename(image_path))[0]
                save_filename = f"{img_basename}_{safe_name}.jpg"
                save_path = os.path.join(args.viz_dir, save_filename)
                cv2.imwrite(save_path, img_with_box)

        except Exception as e:
            error_msg = f"Error processing {row.get('image_path', 'N/A')}: {e}"
            print(f"\n{error_msg}")
            traceback.print_exc()
            results.append(error_msg)

    # 4. Save results
    df['model_feedback'] = results
    output_cols = ['image_path', 'object_name_en', 'object_name_zh', 'ground_truth_prompt', 'model_feedback']
    # Ensure all required columns exist, fill missing ones with empty string
    for col in output_cols:
        if col not in df.columns:
            df[col] = ''

    df_output = df[output_cols]
    df_output.to_excel(args.output_excel, index=False, engine='openpyxl')
    print(f"\nProcessing complete. Results saved to: {args.output_excel}")


def main():
    parser = argparse.ArgumentParser(description="Generate feedback on image quality for object visibility.")

    # --- I/O Arguments ---
    parser.add_argument('--input_excel', type=str, required=True, help="Path to the input Excel file.")
    parser.add_argument('--output_excel', type=str, required=True, help="Path for the output Excel file.")

    # --- Model Selection Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['GroundingDINO', 'YOLOE'],
                        help="Type of detection model to use.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights file (.pth, .pt).")
    parser.add_argument('--config_path', type=str, help="Path to the model config file (required for GroundingDINO).")

    # --- Inference Arguments ---
    parser.add_argument('--box_thresh', type=float, default=0.35, help="Box threshold for GroundingDINO.")
    parser.add_argument('--text_thresh', type=float, default=0.25, help="Text threshold for GroundingDINO.")
    parser.add_argument('--fp16', action='store_true', help="Enable FP16 inference for CUDA devices.")

    # --- Utility Arguments ---
    parser.add_argument('--visualize', action='store_true', help="Save images with bounding boxes and feedback.")
    parser.add_argument('--viz_dir', type=str, default="visualizations", help="Directory to save visualization images.")
    parser.add_argument('--limit', type=int, default=None, help="Limit processing to the first N rows for testing.")

    args = parser.parse_args()
    process_dataset(args)


if __name__ == '__main__':
    main()