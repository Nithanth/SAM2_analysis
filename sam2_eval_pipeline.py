import json
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # Optional: for progress bar
import torch
from PIL import Image

# Import Hugging Face transformers components
# Make sure to install: pip install transformers torch accelerate Pillow
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    # Or replace with specific SAM2 classes if they exist, e.g.:
    # from transformers import SamProcessor, SamModel 
except ImportError:
    print("Error: Could not import transformers library.")
    print("Please install it: pip install transformers torch accelerate Pillow")
    exit()

# Import metric functions from metrics.py
# Ensure metrics.py is in the same directory or accessible via PYTHONPATH
try:
    from metrics import calculate_miou, calculate_boundary_f1
except ImportError:
    print("Error: Could not import metric functions from metrics.py.")
    print("Ensure metrics.py is in the same directory or accessible via PYTHONPATH.")
    exit()

# --- SAM2 Model Interaction using Hugging Face --- 

# Global variables for model and processor to avoid reloading on each call
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam2_model(model_id: str):
    """Loads the SAM2 model and processor from Hugging Face."""
    global model, processor
    if model is not None and processor is not None:
        print("Model and processor already loaded.")
        return model, processor
        
    print(f"Loading SAM2 model and processor for '{model_id}'...")
    try:
        # Replace AutoModelForVision2Seq with the correct class if needed
        model = AutoModelForVision2Seq.from_pretrained(model_id).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Model and processor loaded successfully to {device}.")
        return model, processor
    except Exception as e:
        print(f"Error loading model/processor '{model_id}' from Hugging Face: {e}")
        return None, None

def predict_mask(model, processor, image_path: str) -> tuple[np.ndarray | None, float]:
    """ 
    Runs SAM2 model inference using Hugging Face transformer model.
    Performs automatic mask generation and selects the best mask.
    Returns the binary mask (np.ndarray, 0/1) and its predicted IoU score (float).
    Returns (None, np.nan) on failure.
    """
    # print(f"  Predicting mask for {os.path.basename(image_path)}...")
    try:
        # 1. Load the image using PIL (transformers often prefer PIL)
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Error loading image {image_path}: {e}")
        return None, np.nan
        
    try:
        # 2. Preprocess image and prepare inputs
        # Use processor for automatic mask generation prompt if available
        # The specific prompt might vary depending on the SAM2 version/HF implementation
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        
        # 3. Run model prediction - Disable gradient calculation for inference
        with torch.no_grad():
             # Adjust generation parameters as needed for automatic mask generation
             # Check the specific model's documentation on Hugging Face Hub
            outputs = model.generate(**inputs, max_new_tokens=256) # Example, might need adjustment
            
        # 4. Post-process output to get masks and scores
        # This part is HIGHLY dependent on the specific SAM2 model output structure
        # Check model documentation for how masks and scores are returned
        # Example assumes processor.post_process_segmentation exists and works like this:
        segmentation_result = processor.post_process_segmentation(outputs, target_sizes=[raw_image.size[::-1]])[0]
        
        masks = segmentation_result['masks'] # Assumes boolean [N, H, W] tensor
        scores = segmentation_result['scores'] # Assumes [N] tensor
        
        if masks is None or scores is None or len(masks) == 0:
            print(f"  Warning: No masks generated for {os.path.basename(image_path)}")
            return None, np.nan
            
        # Select the mask with the highest predicted IoU score
        best_idx = torch.argmax(scores).item()
        best_mask_tensor = masks[best_idx] # [H, W] boolean tensor
        best_score = scores[best_idx].item()
        
        # Convert to NumPy array (uint8, 0/1)
        pred_mask_np = best_mask_tensor.cpu().numpy().astype(np.uint8)
        
        # print(f"  Prediction complete. Best score: {best_score:.4f}")
        return pred_mask_np, best_score
        
    except Exception as e:
        print(f"  Error during prediction or post-processing for {image_path}: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed error traceback
        return None, np.nan

# --- End SAM2 Model Interaction Section ---

# --- Utility Functions (load_mask remains the same) ---

def load_mask(path: str) -> np.ndarray:
    """Loads a mask image and converts it to a binary numpy array (0 or 1)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask: {path}")
        return None
    # Convert to binary (0 or 1) - Assumes mask is 0 and non-zero (e.g., 255)
    # Adjust threshold (e.g., 0 or 127) if masks use different values
    _, binary_mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY) # Threshold at 1
    # Alternatively, if mask is already 0/1: binary_mask = mask
    return binary_mask.astype(np.uint8)

# --- Main Pipeline Function --- 

def run_evaluation_pipeline(config: dict):
    """
    Runs the full evaluation pipeline based on the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing configuration parameters:
            - model_id (str): Hugging Face ID of the SAM2 model (e.g., 'facebook/sam2-huge').
            - data_list_path (str): Path to the JSON file listing data items.
            - image_base_dir (str): Base directory containing input images.
            - gt_mask_base_dir (str): Base directory containing ground truth masks.
            - output_path (str): Path to save the resulting CSV file.
            - bf1_tolerance (int, optional): Pixel tolerance for BF1 calc. Defaults to 2.
            - ... (other config params like degradation level can be added)
    """

    print("--- Starting SAM2 Evaluation Pipeline --- ")
    print(f"Configuration: {config}")

    # --- Configuration Validation ---
    required_keys = ['model_id', 'data_list_path', 'image_base_dir', 'gt_mask_base_dir', 'output_path']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key in configuration: '{key}'")
            return

    # --- Load Model (Hugging Face) ---
    global model, processor # Use global vars defined earlier
    # Load model only once if not already loaded
    if model is None or processor is None:
        model, processor = load_sam2_model(config['model_id'])
        if model is None or processor is None: # Check if loading failed
            print("Error: Failed to load model/processor. Exiting.")
            return

    # --- Load Data List ---
    data_list_path = config['data_list_path']
    if not os.path.exists(data_list_path):
        print(f"Error: Data list JSON file not found at {data_list_path}")
        return
    try:
        with open(data_list_path, 'r') as f:
            evaluation_list = json.load(f)
        print(f"Loaded {len(evaluation_list)} items from {data_list_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {data_list_path}: {e}")
        return
    except Exception as e:
        print(f"Error reading data list file {data_list_path}: {e}")
        return


    # --- Parameters ---
    image_base_dir = config.get('image_base_dir', '')
    gt_mask_base_dir = config.get('gt_mask_base_dir', '')
    bf1_tolerance = int(config.get('bf1_tolerance', 2)) # Ensure integer
    results = []

    print(f"Using BF1 tolerance: {bf1_tolerance} pixels")
    print(f"Iterating through {len(evaluation_list)} items...")

    # --- Processing Loop ---
    for item in tqdm(evaluation_list, desc="Evaluating Masks"): # Wrap with tqdm for progress
        image_id = item.get("id", "unknown_id")
        # These paths should be relative within the JSON if base dirs are used
        image_rel_path = item.get("image_path")
        gt_mask_rel_path = item.get("gt_mask_path")
        degradation_info = item.get("degradation", "original") # Get degradation info if present

        if not image_rel_path or not gt_mask_rel_path:
            print(f"Warning: Skipping item ID '{image_id}' due to missing 'image_path' or 'gt_mask_path' in JSON.")
            continue

        # Construct full paths
        image_full_path = os.path.join(image_base_dir, image_rel_path)
        gt_mask_full_path = os.path.join(gt_mask_base_dir, gt_mask_rel_path)

        # Check file existence before proceeding
        if not os.path.isfile(image_full_path):
            print(f"Warning: Skipping item ID '{image_id}'. Image not found: {image_full_path}")
            continue
        if not os.path.isfile(gt_mask_full_path):
            print(f"Warning: Skipping item ID '{image_id}'. Ground truth mask not found: {gt_mask_full_path}")
            continue

        # --- Model Inference (Hugging Face) ---
        pred_mask, sam2_score = predict_mask(model, processor, image_full_path)
        
        if pred_mask is None:
            print(f"Warning: Skipping item ID '{image_id}' due to prediction error.")
            # Optionally record failure with NaN metrics
            results.append({
                "image_id": image_id,
                "degradation": degradation_info,
                "iou": np.nan,
                "bf1": np.nan,
                "sam2_score": np.nan,
                "image_path": image_rel_path,
                "gt_mask_path": gt_mask_rel_path,
                "status": "Prediction Failed"
            })
            continue

        # --- Load Ground Truth Mask ---
        try:
            gt_mask = load_mask(gt_mask_full_path)
            if gt_mask is None:
                print(f"Warning: Skipping item ID '{image_id}' due to GT mask loading error (load_mask returned None).")
                 # Optionally record failure with NaN metrics
                results.append({
                    "image_id": image_id,
                    "degradation": degradation_info,
                    "iou": np.nan,
                    "bf1": np.nan,
                    "sam2_score": sam2_score, # Might have score even if GT fails
                    "image_path": image_rel_path,
                    "gt_mask_path": gt_mask_rel_path,
                    "status": "GT Load Failed"
                })
                continue
        except Exception as e:
             print(f"Warning: Skipping item ID '{image_id}' due to exception during GT mask loading: {e}")
             # Optionally record failure with NaN metrics
             results.append({
                    "image_id": image_id,
                    "degradation": degradation_info,
                    "iou": np.nan,
                    "bf1": np.nan,
                    "sam2_score": sam2_score,
                    "image_path": image_rel_path,
                    "gt_mask_path": gt_mask_rel_path,
                    "status": "GT Load Exception"
                })
             continue

        # --- Calculate Metrics ---
        try:
            # Ensure masks are suitable type (e.g., uint8 0/1 or bool)
            iou = calculate_miou(pred_mask.astype(bool), gt_mask.astype(bool))
            bf1 = calculate_boundary_f1(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8), tolerance_px=bf1_tolerance)
            status = "Success"
        except Exception as e:
            print(f"Warning: Metrics calculation failed for item ID '{image_id}': {e}")
            iou = np.nan
            bf1 = np.nan
            status = "Metrics Calc Exception"
            # sam2_score is already calculated
            
        # --- Store Results ---
        results.append({
            "image_id": image_id,
            "degradation": degradation_info,
            "iou": iou,
            "bf1": bf1,
            "sam2_score": sam2_score, # Already fetched during prediction
            "image_path": image_rel_path, # Store relative paths for reference
            "gt_mask_path": gt_mask_rel_path,
            "status": status
        })

    # --- Aggregate and Save Results ---
    if not results:
        print("Warning: No results were generated or processed.")
        print("--- SAM2 Evaluation Pipeline Finished --- ")
        return

    results_df = pd.DataFrame(results)
    print("\n--- Evaluation Summary --- ")
    successful_results = results_df[results_df['status'] == 'Success']
    num_success = len(successful_results)
    num_total = len(results_df)
    num_failed = num_total - num_success
    
    print(f"Processed {num_total} items. Successful: {num_success}, Failed: {num_failed}")

    if num_success > 0:
        # Calculate average metrics only on successful runs
        mean_iou = successful_results['iou'].mean()
        mean_bf1 = successful_results['bf1'].mean()
        
        print(f"Mean IoU (mIoU) on successful: {mean_iou:.4f}")
        print(f"Mean Boundary F1 (BF1) @ {bf1_tolerance}px on successful: {mean_bf1:.4f}")
        
        # Calculate mean SAM2 score excluding NaNs, only on successful runs
        if successful_results['sam2_score'].notna().any():
            mean_sam2_score = successful_results['sam2_score'].mean(skipna=True)
            print(f"Mean SAM2 Score on successful: {mean_sam2_score:.4f} (calculated over non-NaN values)")
        else:
            print("Mean SAM2 Score on successful: N/A (No scores available or all were NaN)")
    else:
        print("No successful results to calculate average metrics.")

    # Save DataFrame (including failures)
    output_path = config['output_path']
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        results_df.to_csv(output_path, index=False)
        print(f"\nFull results (including failures) saved successfully to: {output_path}")
    except Exception as e:
        print(f"\nError saving results to {output_path}: {e}")

    print("--- SAM2 Evaluation Pipeline Finished --- ")

# Note: This script is intended to be imported and the
# run_evaluation_pipeline function called from another script (e.g., run_experiment.py).
# It does not run the pipeline automatically when executed directly.
# Example of how to call it (in another file):
# from sam2_eval_pipeline import run_evaluation_pipeline
# config = { ... } # Load or define config dictionary
# run_evaluation_pipeline(config)
