import cv2
import torch
import numpy as np
import matplotlib
import sys
import os
import math
import time
import ssl
import pandas as pd
import argparse


# -----------------------------
# NEW: timing helper
# -----------------------------
class TimingRecorder:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        # data[name] = {"total": float, "count": int}
        self.data = {}

    def start(self, name: str):
        if not self.enabled:
            return
        # 用一个隐藏字段存开始时间
        slot = self.data.setdefault(name, {"total": 0.0, "count": 0})
        slot["_t0"] = time.time()

    def end(self, name: str):
        if not self.enabled:
            return
        slot = self.data.get(name, None)
        if not slot:
            return
        t0 = slot.pop("_t0", None)
        if t0 is None:
            return
        dt = time.time() - t0
        slot["total"] += dt
        slot["count"] += 1

    def add(self, name: str, dt: float):
        if not self.enabled:
            return
        slot = self.data.setdefault(name, {"total": 0.0, "count": 0})
        slot["total"] += dt
        slot["count"] += 1

    def print_summary(self):
        if not self.enabled:
            return
        print("\n==================== Timing Summary ====================")
        for name, slot in self.data.items():
            # 跳过还没正常end的
            if name.startswith("_"):
                continue
            total = slot.get("total", 0.0)
            count = slot.get("count", 0)
            if count == 0:
                avg = 0.0
            else:
                avg = total / count
            print(f"{name:20s} | calls: {count:4d} | avg: {avg*1000:.2f} ms")
        print("========================================================\n")


# Import the new GPT helper module
# Ensure 'gpt_helper.py' exists in the same directory.
try:
    import gpt_helper
except ImportError:
    print("Warning: 'gpt_helper.py' not found. GPT-4o functionality will be unavailable.")
    gpt_helper = None

# --- Matplotlib Backend Setup ---
matplotlib.use('Agg')

# --- Global Settings ---
ssl._create_default_https_context = ssl._create_unverified_context
try:
    # Ensure the path to Depth-Anything-V2 is correct
    depth_anything_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Depth-Anything-V2')
    if depth_anything_path not in sys.path:
        sys.path.append(depth_anything_path)
    from depth_anything_v2.dpt import DepthAnythingV2
except (ImportError, FileNotFoundError):
    print(
        "Error: Could not import DepthAnythingV2. Please ensure the 'Depth-Anything-V2' folder is in the same directory as this script.")
    sys.exit(1)

from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

try:
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    print(
        "Warning: 'transformers' library is not installed. To use the Qwen model, please run 'pip install transformers accelerate sentencepiece'")
    Qwen2_5_VLForConditionalGeneration, AutoProcessor = None, None


# =================================================================================
# --- Argument Parser Setup ---
# =================================================================================
def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Process images to identify pointed objects using a multi-model pipeline and LLMs.")

    parser.add_argument('--llm_model_name_or_path', type=str,
                        default="/mnt/data_3/home_aiglasses/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/",
                        help="Name of the OpenAI model (e.g., 'gpt-4o') or the local path to the Qwen model.")

    parser.add_argument('--excel_input_path', type=str,
                        default="/mnt/data_3/home_aiglasses/pointing-dataset/Processed/matched_annotations.xlsx",
                        help="Path to the input Excel file for batch processing.")

    parser.add_argument('--excel_image_dir', type=str,
                        default="/mnt/data_3/home_aiglasses/pointing-dataset/Processed/MatchedSamples/",
                        help="Directory containing the images referenced in the Excel file.")

    parser.add_argument('--excel_output_path', type=str,
                        default="/mnt/data_3/home_aiglasses/pointing-dataset/Processed/ours_answers.xlsx",
                        help="Path to save the output Excel file with results.")

    # -----------------------------
    # NEW: enable detailed timing
    # -----------------------------
    parser.add_argument('--enable_timing', action='store_true',
                        help="Enable detailed timing per stage and print averages at the end.")

    return parser.parse_args()


# =================================================================================
# --- User Configuration Area ---
# =================================================================================

# 1. Processing Mode & Output Configuration
PROCESSING_MODE = 'excel'  # Options: 'list', 'excel'

# 2. Object Matching Strategy
OBJECT_MATCHING_MODE = 'bbox_only'  # Options: 'specific_then_fallback' or 'bbox_only'

# 3. LLM Configuration
LLM_CHOICE = 'qwen'  # <--- Switch between 'openai', 'qwen', or 'none' here
# For OpenAI, it's strongly recommended to use environment variables (export OPENAI_API_KEY=...),
# but you can also fill it in here directly if needed.
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"

# 4. Pointing Analysis & Matching Configuration (All legacy settings preserved)
ADAPTIVE_FINGERTIP_SEARCH_RANGE = (-50, -10)
ADAPTIVE_FINGERTIP_GRADIENT_THRESHOLD = 5.0
FALLBACK_FINGERTIP_OFFSET = -10
PROMPT_FREE_CONF_THRESHOLD = 0.4
MATCHING_DISTANCE_THRESHOLD = 100
USE_ADAPTIVE_THRESHOLD = True
LOCAL_SLOPE_WINDOW_SIZE = 10
SLOPE_HANDLING_MODE = 'positive_only'
BASE_COLLISION_THRESHOLD = 15
ACCUMULATIVE_SLOPE_SCALING = 0.05
INTERSECTION_COOLDOWN_STEPS = 50

USE_ADAPTIVE_FINGERTIP = True
USE_DEPTH_ADAPTIVE_BBOX_SIZE = True
BBOX_BASE_PADDING = 100
BBOX_DEPTH_SCALING_FACTOR = 300
USE_RAY_FOCUS_INSTEAD_OF_INTERSECTION = False  # Set to False to test BBox logic
RAY_FOCUS_WIDTH = 300
SAVE_PREPROCESSED_IMAGE_FOR_MLLM = True

# 5. [NEW] Hyperparameter: Controls the cropping behavior for LLM input images.
# True: If the analysis result is a BBox, crop the image to tightly contain the "hand" and "target" before sending to the LLM.
# False: If the analysis result is a BBox, draw a box on the full original image and send the full image to the LLM (legacy behavior).
CROP_FOR_LLM_WITH_HAND_CONTEXT = True

# [NEW] Hyperparameter: When CROP_FOR_LLM_WITH_HAND_CONTEXT is True, this determines the context source for cropping.
# 'person_bbox': Use the full 'person' bounding box. Wider view, more background context. (Wide Crop)
# 'hand_centroid': Use the tight bounding box of the segmented 'hand/arm'. More focused on the hand action. (Tight Crop)
# 'finger_base': Use the coordinates of the finger base (ROI centroid). Most focused. (Tightest Crop)
# 'interpolated_hand_point': Use a weighted average between 'finger_base' and 'hand_centroid', controlled by the weight below.
CONTEXT_CROP_SOURCE = 'interpolated_hand_point'  # Options: 'person_bbox', 'hand_centroid', 'finger_base', 'interpolated_hand_point'

# [NEW] Weight Hyperparameter: Effective only when CONTEXT_CROP_SOURCE is 'interpolated_hand_point'.
# Range: 0.0 to 1.0
# 0.0 = Purely use finger_base
# 0.5 = Midpoint between finger_base and hand_centroid
# 1.0 = Purely use hand_centroid
HAND_POINT_INTERPOLATION_WEIGHT = 0.25

# [NEW] Directory to save cropped images for debugging and traceability.
CROPPED_IMAGE_OUTPUT_DIR = "llm_cropped_input"

# True: If pointing analysis fails, send the original image and question directly to the LLM for an answer.
# False: If analysis fails, record the failure reason in the results (legacy behavior).
FALLBACK_TO_DIRECT_LLM_ON_FAILURE = True

# 6. Model Path Configuration (Legacy models)
DEPTH_MODEL_ENCODER = 'vitl'
POSE_MODEL_PATH = "yolo11n-pose.pt"
SEG_MODEL_PATH = "yoloe-11l-seg.pt"
YOLOE_PF_MODEL_PATH = "yoloe-11l-seg-pf.pt"

# 7. 'list' mode configuration (for single image testing)
IMAGE_PATHS = ["./2.JPG", "./3.JPG"]
EXCEL_QUESTION_COLUMN = "question"
EXCEL_START_ROW = 3
EXCEL_END_ROW = 100

# 8. Debugging & Visualization
PRINT_TIMING_DETAILS = True
VISUALIZE_RESULTS = True

# =================================================================================
# --- Global Variables & Helper Functions (Logic Preserved) ---
# =================================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
VISUALIZATION_COLORS = [
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 56),
    (255, 56, 255), (56, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]


def get_qwen_response(model, processor, image_np, question):
    """
    Gets a response from the loaded Qwen-VL model for an image and question.

    Args:
        model: The loaded Qwen-VL model.
        processor: The loaded Qwen-VL processor.
        image_np (np.array): Image in BGR format (from cv2.imread).
        question (str): The question to ask the model.

    Returns:
        str: The model's response.
    """
    if model is None or processor is None:
        raise RuntimeError("Qwen model or processor not initialized. Check if 'transformers' is installed.")

    # Qwen requires a PIL Image in RGB format.
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Prepare inputs for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_pil], padding=True, return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response


def create_ray_focus_mask(image_shape, start_point, direction, width):
    """
    Creates a white line-shaped mask on an image representing the "focus area" of a pointing gesture.

    Args:
        image_shape (tuple): The image's shape (h, w) or (h, w, c).
        start_point (np.array): The ray's starting point (x, y, z coordinates).
        direction (np.array): The ray's direction vector (unit vector).
        width (int): The width of the mask line in pixels.

    Returns:
        np.array: A binary mask of the same size as the image, with the focus area as white (255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    start_point_2d = start_point[:2].astype(int)
    direction_2d = direction[:2]

    # Calculate a very distant end point to ensure the line crosses the entire image
    line_length = np.sqrt(h ** 2 + w ** 2) * 1.5
    end_point_2d = (start_point_2d + direction_2d * line_length).astype(int)

    # Draw a thick line on the mask
    cv2.line(mask, tuple(start_point_2d), tuple(end_point_2d), 255, thickness=width)
    return mask


def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0: return 180.0
    angle = math.acos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return math.degrees(angle)


def find_pointing_direction(original_image, hand_mask, depth_map):
    """Analyzes a hand mask to find the fingertip and finger base."""
    orig_h, orig_w = original_image.shape[:2]

    # Ensure mask and depth map match original image dimensions
    hand_mask_resized = cv2.resize(hand_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) if hand_mask.shape[
                                                                                                    :2] != (orig_h,
                                                                                                            orig_w) else hand_mask.copy()
    depth_map_resized = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) if depth_map.shape[
                                                                                                   :2] != (orig_h,
                                                                                                           orig_w) else depth_map

    if hand_mask_resized.dtype != np.uint8: hand_mask_resized = hand_mask_resized.astype(np.uint8)

    # Process mask to find the main hand contour
    kernel = np.ones((5, 5), np.uint8)
    hand_mask_processed = cv2.morphologyEx(hand_mask_resized, cv2.MORPH_CLOSE, kernel)
    hand_mask_processed = cv2.morphologyEx(hand_mask_processed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(hand_mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return original_image, (None, None), None

    hand_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand_contour) < 500: return original_image, (None, None), None

    # Simplify contour and find centroid
    epsilon = 0.005 * cv2.arcLength(hand_contour, True)
    hand_contour = cv2.approxPolyDP(hand_contour, epsilon, True)
    M = cv2.moments(hand_contour)
    if M["m00"] == 0: return original_image, (None, None), None
    hand_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # Find the best candidate for a fingertip based on a scoring system
    image_center = (orig_w // 2, orig_h // 2)
    best_fingertip, max_score = None, -1
    contour_len = len(hand_contour)
    max_dist_from_centroid = max(
        np.linalg.norm(p[0] - hand_centroid) for p in hand_contour) if hand_contour.any() else 0
    max_dist_from_img_center = np.linalg.norm(np.array([0, 0]) - image_center)
    for i in range(contour_len):
        point = hand_contour[i][0]
        score_dist = np.linalg.norm(point - hand_centroid) / (max_dist_from_centroid + 1e-6)
        k = 10
        prev_point = hand_contour[(i - k + contour_len) % contour_len][0]
        next_point = hand_contour[(i + k) % contour_len][0]
        angle = calculate_angle(prev_point, point, next_point)
        score_angle = (180 - angle) / 180.0
        dist_from_center = np.linalg.norm(point - image_center)
        score_center = 1.0 - (dist_from_center / (max_dist_from_img_center + 1e-6))
        total_score = (1.0 * score_dist + 1.0 * score_angle + 1.0 * score_center)
        if total_score > max_score:
            max_score, best_fingertip = total_score, tuple(point)

    raw_fingertip = best_fingertip
    if raw_fingertip is None: return original_image, (None, None), None

    # Refine fingertip location
    roi_radius = int(cv2.arcLength(hand_contour, True) * 0.03)
    roi_mask = np.zeros_like(hand_mask_processed)
    cv2.circle(roi_mask, raw_fingertip, roi_radius, 255, -1)
    finger_segment_mask = cv2.bitwise_and(hand_mask_processed, hand_mask_processed, mask=roi_mask)
    M_roi = cv2.moments(finger_segment_mask)
    if M_roi["m00"] == 0: return original_image, (None, None), None

    roi_centroid = (int(M_roi["m10"] / M_roi["m00"]), int(M_roi["m01"] / M_roi["m00"]))
    vec_to_center = np.array(roi_centroid) - np.array(raw_fingertip)
    norm_vec = vec_to_center / (np.linalg.norm(vec_to_center) + 1e-6)

    # Adaptive fingertip detection using depth gradients
    if USE_ADAPTIVE_FINGERTIP:
        search_start, search_end = ADAPTIVE_FINGERTIP_SEARCH_RANGE
        search_points, depth_profile = [], []
        for offset in range(search_start, search_end):
            p = np.array(raw_fingertip) - norm_vec * offset
            px, py = int(p[0]), int(p[1])
            if 0 <= px < orig_w and 0 <= py < orig_h:
                search_points.append(p.astype(int))
                depth_profile.append(depth_map_resized[py, px])

        if len(depth_profile) > 1:
            depth_gradients = np.abs(np.diff(np.array(depth_profile, dtype=np.float32)))
            if np.max(depth_gradients) > ADAPTIVE_FINGERTIP_GRADIENT_THRESHOLD:
                edge_index = np.argmax(depth_gradients)
                fingertip = tuple(search_points[edge_index])
            else:
                fingertip = tuple((np.array(raw_fingertip) - norm_vec * FALLBACK_FINGERTIP_OFFSET).astype(int))
        else:
            fingertip = tuple((np.array(raw_fingertip) - norm_vec * FALLBACK_FINGERTIP_OFFSET).astype(int))
    else:
        fingertip = tuple((np.array(raw_fingertip) - norm_vec * 10).astype(int))

    finger_base = roi_centroid
    result_image = original_image.copy()
    cv2.circle(result_image, fingertip, 12, (0, 255, 0), -1)  # Final Fingertip
    cv2.circle(result_image, raw_fingertip, 12, (0, 0, 255), 2)  # Raw Fingertip
    cv2.circle(result_image, roi_centroid, 10, (255, 255, 0), -1)  # Finger Base

    return result_image, (fingertip, finger_base), hand_centroid


def find_3d_intersection_point(depth_map, tip_3d, direction_3d, offset_start, step_size=1, max_steps=1000):
    """Traces a 3D ray through the depth map to find intersection points."""
    h, w = depth_map.shape
    current_point = offset_start.copy()
    trajectory, intersections = [current_point.copy()], []
    cooldown_timer, min_pointing_distance = INTERSECTION_COOLDOWN_STEPS, 50
    accumulated_threshold_increment = 0.0

    for i in range(max_steps):
        if cooldown_timer > 0: cooldown_timer -= 1
        current_point += direction_3d * step_size
        trajectory.append(current_point.copy())
        x, y, ray_z = int(current_point[0]), int(current_point[1]), current_point[2]
        if not (0 <= x < w and 0 <= y < h): break
        actual_depth = depth_map[y, x]

        # Adaptive threshold based on local scene slope
        if USE_ADAPTIVE_THRESHOLD and i >= LOCAL_SLOPE_WINDOW_SIZE:
            past_point = trajectory[i - LOCAL_SLOPE_WINDOW_SIZE]
            px, py = int(past_point[0]), int(past_point[1])
            if 0 <= px < w and 0 <= py < h:
                past_depth = depth_map[py, px]
                if past_depth > 0 and actual_depth > 0:
                    slope = -(float(actual_depth) - float(past_depth))
                    if SLOPE_HANDLING_MODE == 'positive_only':
                        accumulated_threshold_increment += max(0, slope)
                    elif SLOPE_HANDLING_MODE == 'absolute':
                        accumulated_threshold_increment += abs(slope)

        collision_depth_threshold = BASE_COLLISION_THRESHOLD + ACCUMULATIVE_SLOPE_SCALING * accumulated_threshold_increment

        is_potential_intersection = abs(ray_z - actual_depth) < collision_depth_threshold
        is_far_enough = np.linalg.norm(current_point - tip_3d) > min_pointing_distance
        if is_potential_intersection and is_far_enough and cooldown_timer == 0:
            intersections.append((x, y, actual_depth))
            cooldown_timer = INTERSECTION_COOLDOWN_STEPS

    return intersections, trajectory


def get_3d_pointing_ray_from_2d_points(points_2d, depth_map):
    """Creates a 3D pointing ray from 2D fingertip and finger base points."""
    fingertip_2d, finger_base_2d = points_2d
    h, w = depth_map.shape
    x_tip, y_tip = np.clip(fingertip_2d[0], 0, w - 1), np.clip(fingertip_2d[1], 0, h - 1)
    x_base, y_base = np.clip(finger_base_2d[0], 0, w - 1), np.clip(finger_base_2d[1], 0, h - 1)

    depth_tip = depth_map[y_tip, x_tip]
    depth_base = depth_map[y_base, x_base]
    if depth_tip == 0 or depth_base == 0: return None, None, None

    tip_3d = np.array([x_tip, y_tip, depth_tip])
    base_3d = np.array([x_base, y_base, depth_base])
    direction_3d = (tip_3d - base_3d).astype(np.float64)
    norm = np.linalg.norm(direction_3d)
    if norm < 1e-6: return None, None, None
    direction_3d /= norm

    return tip_3d, base_3d, direction_3d


def find_pointed_object(image, point_xy, model):
    """Finds the closest detected object to a given 2D point."""
    if PRINT_TIMING_DETAILS: print("\n--- Starting Prompt-Free Object Detection & Matching ---")
    t_start = time.time()
    results = model.predict(image, verbose=False)
    if not results or results[0].boxes is None or results[0].masks is None:
        if PRINT_TIMING_DETAILS: print("   Prompt-Free model did not detect any objects.")
        return None, results

    best_match = None
    min_dist = float('inf')
    for i, box in enumerate(results[0].boxes):
        conf = float(box.conf)
        if conf < PROMPT_FREE_CONF_THRESHOLD: continue

        mask_data = results[0].masks.data[i].cpu().numpy()
        mask_uint8 = (mask_data > 0.5).astype(np.uint8)
        if mask_uint8.shape != image.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        M = cv2.moments(mask_uint8)
        if M["m00"] == 0: continue

        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        distance = np.linalg.norm(np.array(point_xy[:2]) - np.array(centroid))

        if distance < MATCHING_DISTANCE_THRESHOLD and distance < min_dist:
            min_dist = distance
            best_match = {
                'name': model.names[int(box.cls)], 'conf': conf, 'mask': mask_uint8,
                'bbox': box.xyxy[0].cpu().numpy().astype(int), 'distance': distance
            }

    t_end = time.time()
    if best_match:
        print(
            f"   Match successful! Closest object: '{best_match['name']}', Distance: {best_match['distance']:.2f}px. (Time: {t_end - t_start:.3f}s)")
    else:
        print(
            f"   No matching object found within {MATCHING_DISTANCE_THRESHOLD}px radius. (Time: {t_end - t_start:.3f}s)")

    return best_match, results


def create_bbox_from_points(points, padding, image_shape):
    """Creates a bounding box enclosing a list of points with padding."""
    if not points: return None
    points_2d = np.array(points)[:, :2]
    x_min, y_min = np.min(points_2d, axis=0) - padding
    x_max, y_max = np.max(points_2d, axis=0) + padding
    h, w = image_shape[:2]
    return [max(0, int(x_min)), max(0, int(y_min)), min(w, int(x_max)), min(h, int(y_max))]


def visualize_combined_results(image, depth_map, annotated_2d_image, tip_3d, base_3d, intersection_points, trajectory,
                               output_path, final_point_info=None, all_detections_results=None,
                               yoloe_pf_model_names=None):
    """Generates and saves a comprehensive visualization of the processing pipeline."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 12))
    annotated_2d_rgb = cv2.cvtColor(annotated_2d_image, cv2.COLOR_BGR2RGB)

    # Plot 1: 2D Pointing Analysis
    plt.subplot(2, 2, 1)
    plt.imshow(annotated_2d_rgb)
    plt.title('Step 4: 2D Pointing Analysis')
    plt.axis('off')

    # Plot 2: 3D Ray Trajectory on Depth Map
    plt.subplot(2, 2, 2)
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Normalized Depth')
    if trajectory: plt.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'y--', linewidth=1,
                            label='Ray Trajectory')
    if tip_3d is not None: plt.plot(tip_3d[0], tip_3d[1], 'ro', markersize=8, label='3D Fingertip')
    if base_3d is not None: plt.plot(base_3d[0], base_3d[1], 'bo', markersize=8, label='3D Finger Base')
    if intersection_points:
        for i, point in enumerate(intersection_points): plt.plot(point[0], point[1], 'go', markersize=12,
                                                                 label=f'Intersection #{i + 1}' if i == 0 else f"#{i + 1}")
    plt.title('Step 5: Ray Trajectory on Depth Map')
    plt.legend()
    plt.axis('off')

    # Plot 3: Final Result with All Detections
    plt.subplot(2, 2, 3)
    viz_img_bgr, overlay_bgr = image.copy(), image.copy()
    if all_detections_results and all_detections_results[0].masks is not None and yoloe_pf_model_names:
        for i, box in enumerate(all_detections_results[0].boxes):
            if float(box.conf) >= PROMPT_FREE_CONF_THRESHOLD:
                mask_uint8 = (all_detections_results[0].masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
                if mask_uint8.shape != image.shape[:2]: mask_uint8 = cv2.resize(mask_uint8,
                                                                                (image.shape[1], image.shape[0]),
                                                                                interpolation=cv2.INTER_NEAREST)
                color_bgr = VISUALIZATION_COLORS[i % len(VISUALIZATION_COLORS)]
                overlay_bgr[mask_uint8 > 0] = color_bgr
                label = f"{yoloe_pf_model_names[int(box.cls)]} {float(box.conf):.2f}"
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(viz_img_bgr, (x1, y1), (x2, y2), color_bgr, 1)
                cv2.putText(viz_img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
    combined_rgb = cv2.cvtColor(cv2.addWeighted(overlay_bgr, 0.4, viz_img_bgr, 0.6, 0), cv2.COLOR_BGR2RGB)
    plt.imshow(combined_rgb)
    if trajectory:
        traj_arr = np.array(trajectory)
        start_point = traj_arr[0]
        end_point_for_line = intersection_points[-1] if intersection_points else traj_arr[-1]
        plt.plot([start_point[0], end_point_for_line[0]], [start_point[1], end_point_for_line[1]], 'r-', linewidth=2.5,
                 alpha=0.9, label='Pointing Ray')
    if tip_3d is not None: plt.plot(tip_3d[0], tip_3d[1], 'go', markersize=8, label='Ray Start')
    if intersection_points:
        for i, point in enumerate(intersection_points): plt.plot(point[0], point[1], 'gx', markersize=15,
                                                                 markeredgewidth=3,
                                                                 label=f'Final Intersection #{i + 1}' if i == 0 else f"#{i + 1}")
    if final_point_info:
        if final_point_info['type'] == 'mask':
            obj = final_point_info['data']
            x1, y1, x2, y2 = obj['bbox']
            label_text = f"Matched: {obj['name']} ({obj['conf']:.2f})"
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', facecolor='none', linewidth=3,
                              linestyle='-', label=label_text))
        elif final_point_info['type'] == 'bbox':
            x1, y1, x2, y2 = final_point_info['data']
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='cyan', facecolor='none', linewidth=2,
                              linestyle='--', label='Fallback BBox for LLM'))
    plt.title('Final Result: Pointing & All Detections')
    plt.legend()
    plt.axis('off')

    # Plot 4: Depth Profile Debug View
    plt.subplot(2, 2, 4)
    if trajectory:
        traj_arr = np.array(trajectory)
        steps = np.arange(len(traj_arr))
        ray_depths = traj_arr[:, 2]
        scene_depths = [
            depth_map[np.clip(int(p[1]), 0, depth_map.shape[0] - 1), np.clip(int(p[0]), 0, depth_map.shape[1] - 1)] for
            p in traj_arr]
        plt.plot(steps, ray_depths, 'b-', label='Ray Depth')
        plt.plot(steps, scene_depths, 'g--', label='Scene Depth')
        if intersection_points:
            for i, point in enumerate(intersection_points):
                intersect_idx = np.argmin([np.linalg.norm(p - point) for p in traj_arr])
                plt.axvline(x=intersect_idx, color='r', linestyle='--',
                            label=f'Intersection #{i + 1}' if i == 0 else f"#{i + 1}")
                plt.axhline(y=point[2], color='r', linestyle='-.', alpha=0.6)
    plt.xlabel('Steps along Ray')
    plt.ylabel('Depth Value')
    plt.title('Debug: Depth Profile')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    if PRINT_TIMING_DETAILS: print(f"\n[+] Unified result image saved to '{output_path}'")


# =================================================================================
# --- Core Processing Pipeline ---
# =================================================================================

def process_image(image_path, depth_model, pose_model, yoloe_model, yoloe_pf_model, timing_recorder=None):
    """
    加了 timing_recorder，用来记录各阶段耗时
    """
    try:
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise FileNotFoundError(f"Failed to read: {image_path}")
    except Exception as e:
        return {'success': False, 'type': 'error', 'data': str(e)}

    try:
        # Step 1: Depth Estimation
        if timing_recorder: timing_recorder.start("depth_estimation")
        depth_raw = depth_model.infer_image(raw_img)
        norm_depth = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
        depth_map_final = (norm_depth * 255.0).astype(np.uint8)
        if timing_recorder: timing_recorder.end("depth_estimation")

        # Step 2: Pose Detection
        if timing_recorder: timing_recorder.start("pose_detection")
        pose_results = pose_model.predict(raw_img, verbose=False)
        if timing_recorder: timing_recorder.end("pose_detection")
        if not pose_results or pose_results[0].boxes is None:
            return {'success': False, 'type': 'error', 'data': "Pose detection failed"}
        detected_person_bboxes = [b.xyxy[0].cpu().numpy() for b in pose_results[0].boxes if int(b.cls[0]) == 0]
        if not detected_person_bboxes:
            return {'success': False, 'type': 'error', 'data': "No person detected"}
        person_bbox = detected_person_bboxes[0]

        # Step 3: YOLOE Segmentation for Hand
        if timing_recorder: timing_recorder.start("hand_segmentation")
        visual_prompts = dict(bboxes=np.array([person_bbox]), cls=np.zeros(1, dtype=int))
        yoloe_results = yoloe_model.predict(raw_img, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor,
                                            verbose=False)
        if timing_recorder: timing_recorder.end("hand_segmentation")
        if not (yoloe_results and yoloe_results[0].masks is not None and len(yoloe_results[0].masks) > 0):
            return {'success': False, 'type': 'error', 'data': "YOLOE hand segmentation failed"}
        person_mask = (yoloe_results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)

        # Steps 4 & 5: Pointing Analysis & 3D Ray Casting
        if timing_recorder: timing_recorder.start("pointing_2d_3d")
        annotated_2d_image, points_2d, hand_centroid = find_pointing_direction(raw_img, person_mask, depth_map_final)
        if points_2d[0] is None:
            if timing_recorder: timing_recorder.end("pointing_2d_3d")
            return {'success': False, 'type': 'error', 'data': "Could not find pointing finger"}
        fingertip_2d, finger_base_2d = points_2d
        tip_3d, base_3d, direction_3d = get_3d_pointing_ray_from_2d_points(points_2d, depth_map_final)
        if timing_recorder: timing_recorder.end("pointing_2d_3d")
        if tip_3d is None:
            return {'success': False, 'type': 'error', 'data': "Could not create 3D ray"}

    except Exception as e:
        return {'success': False, 'type': 'error', 'data': f"Exception during image processing: {e}"}

    final_point_info = None
    all_pf_results = None

    if USE_RAY_FOCUS_INSTEAD_OF_INTERSECTION:
        # ray focus 的分支不细拆了
        focus_mask = create_ray_focus_mask(raw_img.shape, tip_3d, direction_3d, RAY_FOCUS_WIDTH)
        image_for_llm = raw_img.copy()
        dark_overlay = cv2.addWeighted(image_for_llm, 0.2, np.zeros_like(image_for_llm), 0.8, 0)
        image_for_llm[focus_mask == 0] = dark_overlay[focus_mask == 0]
        if SAVE_PREPROCESSED_IMAGE_FOR_MLLM:
            base, _ = os.path.splitext(os.path.basename(image_path))
            os.makedirs("preprocessed_for_llm", exist_ok=True)
            save_path = os.path.join("preprocessed_for_llm", f"{base}_mllm_input.png")
            cv2.imwrite(save_path, image_for_llm)
        _, all_pf_results = find_pointed_object(raw_img, (0, 0), yoloe_pf_model)
        intersection_points = []
        trajectory = [tip_3d, tip_3d + direction_3d * 2000]
        result_dict = {'success': True, 'type': 'image_for_llm', 'data': image_for_llm}
    else:
        print("\n--- [3D Intersection Mode] Analyzing ray-scene collision ---")
        offset_start = tip_3d + direction_3d * 30

        if timing_recorder: timing_recorder.start("ray_intersection")
        intersection_points, trajectory = find_3d_intersection_point(depth_map_final, tip_3d, direction_3d,
                                                                     offset_start)
        if timing_recorder: timing_recorder.end("ray_intersection")

        if intersection_points:
            padding_to_use = 40
            if USE_DEPTH_ADAPTIVE_BBOX_SIZE:
                avg_depth = np.mean(np.array(intersection_points)[:, 2])
                farness_factor = (255.0 - avg_depth) / 255.0
                dynamic_padding = BBOX_BASE_PADDING + farness_factor * BBOX_DEPTH_SCALING_FACTOR
                padding_to_use = int(dynamic_padding)
                print(
                    f"  -> [Depth Adaptive BBox] Avg Depth: {avg_depth:.2f}, Farness Factor: {farness_factor:.2f}, Calculated Padding: {padding_to_use}")

            if OBJECT_MATCHING_MODE == 'specific_then_fallback':
                if timing_recorder: timing_recorder.start("object_matching")
                primary_intersection = intersection_points[0]
                matched_object, all_pf_results = find_pointed_object(raw_img, primary_intersection, yoloe_pf_model)
                if timing_recorder: timing_recorder.end("object_matching")
                if matched_object:
                    result_dict = {'success': True, 'type': 'mask', 'data': matched_object}
                    final_point_info = result_dict
                else:
                    fallback_bbox = create_bbox_from_points(intersection_points, padding_to_use, raw_img.shape)
                    if fallback_bbox:
                        result_dict = {'success': True, 'type': 'bbox', 'data': fallback_bbox}
                        final_point_info = {'type': 'bbox', 'data': fallback_bbox}
                    else:
                        result_dict = {'success': False, 'type': 'error', 'data': "Failed to generate fallback BBox"}

            elif OBJECT_MATCHING_MODE == 'bbox_only':
                target_bbox = create_bbox_from_points(intersection_points, padding_to_use, raw_img.shape)
                if target_bbox:
                    result_dict = {
                        'success': True,
                        'type': 'bbox',
                        'data': {
                            'target_bbox': target_bbox,
                            'person_bbox': person_bbox.astype(int),
                            'hand_centroid': hand_centroid,
                            'finger_base_2d': finger_base_2d
                        }
                    }
                    final_point_info = {'type': 'bbox', 'data': target_bbox}
                else:
                    result_dict = {'success': False, 'type': 'error', 'data': "Failed to generate BBox"}
        else:
            result_dict = {'success': False, 'type': 'error', 'data': "Ray did not intersect with any object"}

    # Visualization
    if VISUALIZE_RESULTS and result_dict['success']:
        if timing_recorder: timing_recorder.start("visualization")
        base, _ = os.path.splitext(os.path.basename(image_path))
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)
        output_viz_path = os.path.join(output_dir, f"{base}_result.png")
        visualize_combined_results(
            raw_img, depth_map_final, annotated_2d_image, tip_3d, base_3d,
            intersection_points, trajectory, output_viz_path,
            final_point_info=final_point_info,
            all_detections_results=all_pf_results,
            yoloe_pf_model_names=yoloe_pf_model.names if yoloe_pf_model else None
        )
        if timing_recorder: timing_recorder.end("visualization")

    return result_dict

def process_and_query_llm(image_path, analysis_result, question, llm_objects, llm_model_name_or_path,
                          timing_recorder=None):
    """
    Handles image preparation (cropping/drawing) and querying the selected LLM.
    """
    final_answer = ""
    qwen_model, qwen_processor = llm_objects.get('qwen', (None, None))

    # 开始计时
    if timing_recorder:
        timing_recorder.start("llm_processing")

    if not analysis_result['success']:
        failure_reason = analysis_result['data']
        print(f"  -> Pointing analysis failed: {failure_reason}")
        if FALLBACK_TO_DIRECT_LLM_ON_FAILURE and LLM_CHOICE != 'none':
            print("  -> [Fallback Triggered] Preparing to query LLM on the original image...")
            try:
                original_image = cv2.imread(image_path)
                if original_image is None:
                    return f"Fallback_Error: Could not re-read image for fallback."

                llm_response = ""
                if LLM_CHOICE == 'openai':
                    if not gpt_helper: return "LLM_Error: gpt_helper not available."
                    print("     Calling OpenAI GPT-4o for fallback analysis...")
                    llm_response = gpt_helper.get_gpt4o_response(original_image, question, llm_model_name_or_path)
                elif LLM_CHOICE == 'qwen':
                    print("     Calling Qwen-VL for fallback analysis...")
                    llm_response = get_qwen_response(qwen_model, qwen_processor, original_image, question)

                final_answer = f"Fallback_LLM: {llm_response}"

            except Exception as e:
                final_answer = f"Fallback_LLM_Error: {e}"
        else:
            final_answer = f"Analysis_Failed: {failure_reason}"

    elif analysis_result['type'] == 'mask':
        final_answer = f"Object_Matched: {analysis_result['data']['name']} (Confidence: {analysis_result['data']['conf']:.2f})"

    elif analysis_result['type'] in ['bbox', 'image_for_llm']:
        if LLM_CHOICE == 'none':
            return f"BBox_Generated (LLM not used): {analysis_result['data']}"

        print(f"  -> Analysis successful. Preparing image for {LLM_CHOICE.upper()}...")
        try:
            original_image = cv2.imread(image_path)
            image_for_llm = None

            # This branch handles the legacy 'ray focus' mode
            if analysis_result['type'] == 'image_for_llm':
                image_for_llm = analysis_result['data']

            # This is the primary branch for 'bbox' results
            elif analysis_result['type'] == 'bbox':
                bbox_data = analysis_result['data']
                target_bbox = bbox_data['target_bbox']

                # --- [NEW] Cropping logic branch ---
                if CROP_FOR_LLM_WITH_HAND_CONTEXT:
                    context_point = None
                    # Strategy: Interpolated Point
                    if CONTEXT_CROP_SOURCE == 'interpolated_hand_point' and 'finger_base_2d' in bbox_data and 'hand_centroid' in bbox_data:
                        print(
                            f"  -> [Crop Mode] Using 'interpolated_hand_point' (Weight: {HAND_POINT_INTERPOLATION_WEIGHT}).")
                        weight = HAND_POINT_INTERPOLATION_WEIGHT
                        finger_base = np.array(bbox_data['finger_base_2d'])
                        hand_centroid = np.array(bbox_data['hand_centroid'])
                        context_point = ((1 - weight) * finger_base + weight * hand_centroid).astype(int)
                    # Strategy: Other Point-based sources
                    elif CONTEXT_CROP_SOURCE == 'hand_centroid' and 'hand_centroid' in bbox_data:
                        print("  -> [Crop Mode] Using 'hand_centroid'.")
                        context_point = bbox_data['hand_centroid']
                    elif CONTEXT_CROP_SOURCE == 'finger_base' and 'finger_base_2d' in bbox_data:
                        print("  -> [Crop Mode] Using 'finger_base'.")
                        context_point = bbox_data['finger_base_2d']

                    # If point-based crop is chosen and valid, calculate combined bbox
                    if context_point is not None:
                        x1 = min(target_bbox[0], context_point[0])
                        y1 = min(target_bbox[1], context_point[1])
                        x2 = max(target_bbox[2], context_point[0])
                        y2 = max(target_bbox[3], context_point[1])
                    # Strategy: BBox-based crop (or fallback)
                    else:
                        print(f"  -> [Crop Mode] Using 'person_bbox' (or fallback from '{CONTEXT_CROP_SOURCE}').")
                        context_bbox = bbox_data['person_bbox']
                        x1 = min(target_bbox[0], context_bbox[0])
                        y1 = min(target_bbox[1], context_bbox[1])
                        x2 = max(target_bbox[2], context_bbox[2])
                        y2 = max(target_bbox[3], context_bbox[3])

                    # Perform the crop with padding
                    h, w = original_image.shape[:2]
                    padding = 15
                    y1, y2 = max(0, y1 - padding), min(h, y2 + padding)
                    x1, x2 = max(0, x1 - padding), min(w, x2 + padding)

                    cropped_image = original_image[y1:y2, x1:x2]
                    image_for_llm = cropped_image
                    print(f"     - Cropped to size: {cropped_image.shape[1]}x{cropped_image.shape[0]}px")

                    # Save the cropped image for debugging
                    os.makedirs(CROPPED_IMAGE_OUTPUT_DIR, exist_ok=True)
                    base, _ = os.path.splitext(os.path.basename(image_path))
                    save_path = os.path.join(CROPPED_IMAGE_OUTPUT_DIR, f"{base}_cropped.png")
                    cv2.imwrite(save_path, cropped_image)
                    print(f"     - Cropped image saved to: {save_path}")

                # --- [LEGACY] Drawing logic branch ---
                else:
                    print("  -> [Draw Mode] Drawing target BBox on original image...")
                    if not gpt_helper: return "LLM_Error: gpt_helper not available for drawing."
                    image_with_bbox = gpt_helper.draw_bbox_on_image(original_image, target_bbox)
                    image_for_llm = image_with_bbox

            # --- Query the selected LLM ---
            if LLM_CHOICE == 'openai':
                if not gpt_helper: return "LLM_Error: gpt_helper not available."
                print(f"  -> Querying OpenAI {llm_model_name_or_path}...")
                llm_response = gpt_helper.get_gpt4o_response(image_for_llm, question, llm_model_name_or_path)
                final_answer = llm_response
            elif LLM_CHOICE == 'qwen':
                print(f"  -> Querying local Qwen-VL model...")
                llm_response = get_qwen_response(qwen_model, qwen_processor, image_for_llm, question)
                final_answer = llm_response

        except Exception as e:
            final_answer = f"LLM_Processing_Error: {e}"

    if timing_recorder:
        timing_recorder.end("llm_processing")

    return final_answer


def run_excel_processing(args, depth_model, pose_model, yoloe_model, yoloe_pf_model, llm_objects, timing_recorder=None):
    """Runs the full pipeline in batch mode on an Excel file."""
    try:
        df = pd.read_excel(args.excel_input_path, header=None)
    except FileNotFoundError:
        print(f"❌ Error: Excel file not found at {args.excel_input_path}")
        return

    final_results = []
    # start_idx = EXCEL_START_ROW - 1
    # end_idx = min(EXCEL_END_ROW, len(df))
    #
    # for idx in range(start_idx, end_idx):
    #     row = df.iloc[idx]
    #     try:
    #         case_id = str(int(row[0]))
    #         question = str(row[1])
    #         image_name = f"{case_id}.jpg"
    #     except (IndexError, ValueError, TypeError):
    #         print(f"  -> Warning: Skipping row {idx + 1} due to incorrect data format.")
    #         continue

    # --- 使用新的、更健壮的循环方式 ---
    # df.iterrows() 会遍历DataFrame中的每一行
    # idx 是行号 (0, 1, 2, ...), row 是该行的数据 (一个Series对象)
    for idx, row in df.iterrows():
        try:
            # 直接从row中按列索引取值，并且不再需要int()转换
            case_id = str(row[0])
            question = str(row[1])
            image_name = f"{case_id}.jpg"  # 这将正确生成 "pointing_0.jpg"
        except (IndexError, KeyError):
            print(f"  -> Warning: Skipping row {idx + 1} due to incorrect data format or missing columns.")
            continue
        image_path = os.path.join(args.excel_image_dir, image_name)
        print(f"\n{'=' * 20} Processing Sample [{idx + 1}/{len(df)}] | ID: {case_id} {'=' * 20}")

        if not os.path.exists(image_path):
            final_results.append([case_id, question, "Image not found"])
            print(f"  -> Error: Image not found at {image_path}")
            continue

        analysis_result = process_image(image_path, depth_model, pose_model, yoloe_model, yoloe_pf_model,
                                        timing_recorder=timing_recorder)

        final_answer = process_and_query_llm(
            image_path, analysis_result, question, llm_objects, args.llm_model_name_or_path,
            timing_recorder=timing_recorder
        )

        print(f"  -> Final Answer: {final_answer}")
        final_results.append([case_id, question, final_answer])

    # Save results to a new Excel file
    df_out = pd.DataFrame(final_results, columns=["Case_ID", "Question", "LLM_Answer"])
    output_dir = os.path.dirname(args.excel_output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    df_out.to_excel(args.excel_output_path, index=False)
    print(f"\n🎉 Batch processing complete! Results saved to: {args.excel_output_path}")

    # 结束后统一打印
    if timing_recorder and timing_recorder.enabled:
        timing_recorder.print_summary()


def run_list_processing(args, depth_model, pose_model, yoloe_model, yoloe_pf_model, llm_objects, timing_recorder=None):
    """Runs the pipeline on a predefined list of images for testing."""
    for image_path in IMAGE_PATHS:
        question = "What am I pointing at?"  # Example question
        case_id = os.path.splitext(os.path.basename(image_path))[0]

        print(f"\n{'=' * 20} Processing Image | Path: {image_path} {'=' * 20}")

        if not os.path.exists(image_path):
            print(f"  -> Error: Image not found at {image_path}")
            continue

        analysis_result = process_image(image_path, depth_model, pose_model, yoloe_model, yoloe_pf_model)

        final_answer = process_and_query_llm(
            image_path, analysis_result, question, llm_objects, args.llm_model_name_or_path
        )

        print(f"  -> Final Answer: {final_answer}")

    print("\n🎉 List processing complete!")
    if timing_recorder and timing_recorder.enabled:
        timing_recorder.print_summary()

def main():
    """Main function to load models and start the processing workflow."""
    args = parse_arguments()

    # NEW: init timing recorder
    timing_recorder = TimingRecorder(enabled=args.enable_timing)

    # --- 1. Initialize & Load Models ---
    print("=" * 50)
    print("INITIALIZING AND LOADING MODELS")
    print("=" * 50)
    llm_objects = {}

    if LLM_CHOICE == 'openai':
        if not gpt_helper:
            print("❌ Error: gpt_helper module is required for OpenAI but could not be imported.")
            return
        if not OPENAI_API_KEY or "YOUR_OPENAI_API_KEY_HERE" in OPENAI_API_KEY:
            print(
                "❌ Error: OpenAI API Key is not set. Please set the OPENAI_API_KEY in the script or as an environment variable.")
            return
        gpt_helper.initialize_client(OPENAI_API_KEY)
        print("--- OpenAI client initialized. Using model:", args.llm_model_name_or_path)

    elif LLM_CHOICE == 'qwen':
        if not Qwen2_5_VLForConditionalGeneration:
            print("❌ Error: 'transformers' library is required for Qwen but could not be imported.")
            return
        print(f"--- Loading Qwen-VL model from: '{args.llm_model_name_or_path}' ---")
        print("This may take some time and consume significant VRAM...")
        try:
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.llm_model_name_or_path,
                torch_dtype="auto",
                device_map="auto"
            )
            qwen_processor = AutoProcessor.from_pretrained(args.llm_model_name_or_path)
            llm_objects['qwen'] = (qwen_model, qwen_processor)
            print("--- Qwen-VL model loaded successfully ---")
        except Exception as e:
            print(f"❌ Error loading Qwen model: {e}")
            sys.exit(1)
    else:
        print("--- LLM is disabled (LLM_CHOICE = 'none') ---")

    print("\n--- Loading all local vision models... ---")
    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}
    depth_model = DepthAnythingV2(**model_configs[DEPTH_MODEL_ENCODER])
    depth_model_path = f'./Depth-Anything-V2/checkpoints/depth_anything_v2_{DEPTH_MODEL_ENCODER}.pth'
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    print(f"  > Depth model loaded: {depth_model_path}")

    pose_model = YOLO(POSE_MODEL_PATH)
    print(f"  > Pose model loaded: {POSE_MODEL_PATH}")

    yoloe_model = YOLOE(SEG_MODEL_PATH)
    print(f"  > YOLOE Seg model loaded: {SEG_MODEL_PATH}")

    yoloe_pf_model = YOLOE(YOLOE_PF_MODEL_PATH)
    print(f"  > YOLOE Prompt-Free model loaded: {YOLOE_PF_MODEL_PATH}")

    print("--- All local models loaded successfully ---")

    # --- 2. Execute Processing Based on Mode ---
    if PROCESSING_MODE == 'excel':
        run_excel_processing(args, depth_model, pose_model, yoloe_model, yoloe_pf_model, llm_objects,
                             timing_recorder=timing_recorder)
    elif PROCESSING_MODE == 'list':
        run_list_processing(args, depth_model, pose_model, yoloe_model, yoloe_pf_model, llm_objects,
                            timing_recorder=timing_recorder)
    else:
        print(f"❌ Error: Invalid PROCESSING_MODE '{PROCESSING_MODE}'. Please choose 'excel' or 'list'.")


if __name__ == "__main__":
    main()