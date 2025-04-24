import json
import os
import cv2 # Used for loading images as NumPy arrays
import numpy as np
import pandas as pd
from tqdm import tqdm # Optional: for progress bar
import torch
# PIL is not strictly needed if loading with cv2

# Import components from the official sam2 library
# IMPORTANT: You must install the sam2 library first!
# (e.g., git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .)
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # Assuming automatic mask generation uses a class similar to original SAM
    # Check automatic_mask_generator_example.ipynb for the correct import and usage!
    from sam2.automatic_mask_generator import SamAutomaticMaskGenerator 
except ImportError:
    print("Error: Could not import from the 'sam2' library.")
    print("Please ensure you have cloned the repo and installed it following the official README:")
    print("  git clone https://github.com/facebookresearch/sam2.git")
    print("  cd sam2")
    print("  pip install -e .")
    print("  cd ..")
    exit()

# Import metric functions from metrics.py
# Ensure metrics.py is in the same directory or accessible via PYTHONPATH
try:
    from metrics import calculate_miou, calculate_boundary_f1
except ImportError:
    print("Error: Could not import metric functions from metrics.py.")
    print("Ensure metrics.py is in the same directory or accessible via PYTHONPATH.")
    exit()

# --- SAM2 Model Interaction using sam2 library + HF Hub --- 

# Global variables for predictor and generator to avoid reloading
predictor = None
mask_generator = None # For automatic mask generation
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam2_predictor_and_generator(model_hf_id: str, generator_config: dict = None):
    """Loads the SAM2 predictor using .from_pretrained and initializes the automatic mask generator."""
    global predictor, mask_generator
    if predictor is not None and mask_generator is not None:
        print("Predictor and Mask Generator already loaded.")
        return predictor, mask_generator
        
    print(f"Loading SAM2 predictor from Hugging Face ID: '{model_hf_id}'...")
    try:
        # Load the predictor using the sam2 library's HF integration
        predictor = SAM2ImagePredictor.from_pretrained(model_hf_id)
        predictor.model.to(device) # Ensure model is on the correct device
        print(f"Predictor loaded successfully to {device}.")

        # Initialize the Automatic Mask Generator
        # Default config example; adjust based on automatic_mask_generator_example.ipynb
        if generator_config is None:
             # Example settings - **ADJUST THESE BASED ON THE SAM2 EXAMPLE NOTEBOOK**
            generator_config = {
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "crop_n_layers": 0, # 0 means no cropping - faster for single images
                "crop_n_points_downscale_factor": 1,
                "min_mask_region_area": 100, # Minimum mask area in pixels
            }
        print(f"Initializing SamAutomaticMaskGenerator with config: {generator_config}")
        mask_generator = SamAutomaticMaskGenerator(predictor.model, **generator_config)
        print("SamAutomaticMaskGenerator initialized.")

        return predictor, mask_generator
    except Exception as e:
        print(f"Error loading predictor '{model_hf_id}' or initializing generator: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed error traceback
        return None, None

def predict_auto_mask(mask_generator, image_path: str) -> tuple[np.ndarray | None, float]:
    """ 
    Generates masks automatically using SamAutomaticMaskGenerator.
    Selects the 'best' mask based on predicted IoU.
    Returns the binary mask (np.ndarray, 0/1) and its predicted IoU score (float).
    Returns (None, np.nan) on failure.
    """
    # print(f"  Generating masks for {os.path.basename(image_path)}...")
    try:
        # 1. Load the image using OpenCV (usually expected as HWC BGR NumPy array)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"  Error loading image {image_path} with OpenCV")
            return None, np.nan
        # Convert to RGB for the model if needed (check mask_generator requirements)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"  Error loading/processing image {image_path}: {e}")
        return None, np.nan
        
    try:
        # 2. Generate masks
        # Ensure model is in eval mode and use appropriate context managers
        mask_generator.model.eval()
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16): # Use bfloat16 as per README
            # The output 'masks' is typically a list of dictionaries
            generated_data = mask_generator.generate(image_rgb)

        if not generated_data:
            print(f"  Warning: No masks generated for {os.path.basename(image_path)}")
            return None, np.nan

        # 3. Select the best mask (e.g., highest predicted IoU)
        # Sort masks by 'predicted_iou' in descending order
        sorted_masks = sorted(generated_data, key=lambda x: x['predicted_iou'], reverse=True)
        best_mask_info = sorted_masks[0]
        
        best_mask_np = best_mask_info['segmentation'] # This should be a HxW boolean NumPy array
        best_score = best_mask_info['predicted_iou']

        # Convert boolean mask to uint8 (0/1)
        pred_mask_out = best_mask_np.astype(np.uint8)

        # print(f"  Mask generation complete. Best score: {best_score:.4f}")
        return pred_mask_out, best_score
        
    except Exception as e:
        print(f"  Error during automatic mask generation for {image_path}: {e}")
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
    return binary_mask.astype(np.uint8)

# --- Main Pipeline Function --- 

def run_evaluation_pipeline(config: dict):
    """
    Runs the full evaluation pipeline based on the provided configuration dictionary,
    using the official sam2 library loaded via Hugging Face ID.

    Args:
        config (dict): A dictionary containing configuration parameters:
            - model_hf_id (str): Hugging Face ID of the SAM2 model (e.g., 'facebook/sam2-hiera-large').
            - data_list_path (str): Path to the JSON file listing data items.
            - image_base_dir (str): Base directory containing input images.
            - gt_mask_base_dir (str): Base directory containing ground truth masks.
            - output_path (str): Path to save the resulting CSV file.
            - bf1_tolerance (int, optional): Pixel tolerance for BF1 calc. Defaults to 2.
            - generator_config (dict, optional): Configuration for SamAutomaticMaskGenerator.
                                                Overrides defaults if provided.
            - ... (other config params like degradation level can be added)
    """

    print("--- Starting SAM2 Evaluation Pipeline (using sam2 library + HF Hub) --- ")
    print(f"Configuration: {config}")

    # --- Configuration Validation ---
    # Ensure model_hf_id is present instead of model_id/model_path
    required_keys = ['model_hf_id', 'data_list_path', 'image_base_dir', 'gt_mask_base_dir', 'output_path']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key in configuration: '{key}'")
            return

    # --- Load Model & Generator ---
    global predictor, mask_generator # Use global vars defined earlier
    # Load only once if not already loaded
    if predictor is None or mask_generator is None:
        predictor, mask_generator = load_sam2_predictor_and_generator(
            config['model_hf_id'], 
            config.get('generator_config') # Pass optional generator config
        )
        if predictor is None or mask_generator is None: # Check if loading failed
            print("Error: Failed to load predictor or initialize generator. Exiting.")
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

        # --- Model Inference (Automatic Mask Generation) ---
        pred_mask, sam2_score = predict_auto_mask(mask_generator, image_full_path)
        
        if pred_mask is None:
            print(f"Warning: Skipping item ID '{image_id}' due to mask generation error.")
            # Record failure 
            results.append({
                "image_id": image_id,
                "degradation": degradation_info,
                "iou": np.nan,
                "bf1": np.nan,
                "sam2_score": np.nan,
                "image_path": image_rel_path,
                "gt_mask_path": gt_mask_rel_path,
                "status": "Mask Generation Failed"
            })
            continue

        # --- Load Ground Truth Mask ---
        try:
            gt_mask = load_mask(gt_mask_full_path)
            if gt_mask is None:
                print(f"Warning: Skipping item ID '{image_id}' due to GT mask loading error (load_mask returned None).")
                 # Record failure
                results.append({
                    "image_id": image_id,
                    "degradation": degradation_info,
                    "iou": np.nan,
                    "bf1": np.nan,
                    "sam2_score": sam2_score, # Score is from prediction step
                    "image_path": image_rel_path,
                    "gt_mask_path": gt_mask_rel_path,
                    "status": "GT Load Failed"
                })
                continue
        except Exception as e:
             print(f"Warning: Skipping item ID '{image_id}' due to exception during GT mask loading: {e}")
             # Record failure
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
            # Ensure masks are uint8 (0/1)
            iou = calculate_miou(pred_mask.astype(bool), gt_mask.astype(bool))
            bf1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance_px=bf1_tolerance)
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
            "sam2_score": sam2_score, # From prediction step
            "image_path": image_rel_path, 
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
# config = { 'model_hf_id': 'facebook/sam2-hiera-large', ... } # Load or define config dictionary
# run_evaluation_pipeline(config)
