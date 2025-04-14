import numpy as np
import cv2
import pandas as pd
import torch # Added torch import
import os # Added os import

# +++ Add SAM2 Imports +++
try:
    # Assuming SAM2 installed via: pip install -e ./external/sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # Uncomment if using 'auto' prompt type later
    # from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"--- WARNING: Error importing SAM2: {e} ---")
    print("Ensure SAM2 is installed correctly from the submodule:")
    print("  1. cd external/sam2")
    print("  2. pip install -e .")
    print("  3. cd ../..")
    print("Evaluation functions requiring SAM2 will fail.")
    build_sam2 = None
    SAM2ImagePredictor = None

# +++ Update SAM2 Model Loader for Image Predictor +++
def load_sam2_image_predictor(checkpoint_path, config_path, device='cuda'):
    """
    Load SAM2 model and wrap it in an Image Predictor.

    Args:
        checkpoint_path (str): Path to the SAM2 model checkpoint (.pt file).
        config_path (str): Path to the SAM2 model configuration file (.yaml).
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        Loaded SAM2ImagePredictor object.

    Raises:
        ImportError: If SAM2 components could not be imported.
        FileNotFoundError: If checkpoint or config file not found.
        Exception: For other model loading errors.
    """
    if build_sam2 is None or SAM2ImagePredictor is None:
        raise ImportError("SAM2 components could not be imported. Cannot load predictor.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM2 Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
         raise FileNotFoundError(f"SAM2 Config not found: {config_path}")

    print(f"Loading SAM2 from checkpoint: {checkpoint_path}, config: {config_path} onto device: {device}")
    try:
        # Note: Parameter names 'cfg_path' and 'ckpt_path' might need adjustment
        # based on the exact signature of build_sam2 in your SAM2 version.
        # Verify these in the SAM2 library source if errors occur.
        sam2_model = build_sam2(cfg_path=config_path, ckpt_path=checkpoint_path, device=device)
        sam2_model.eval() # Set model to evaluation mode
        predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 model loaded successfully and wrapped in Image Predictor.")
        return predictor # Return the predictor instance
    except Exception as e:
        print(f"--- ERROR loading SAM2 model/predictor: {e} ---")
        raise


def calculate_miou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between a predicted binary mask
    and a ground truth binary mask. Robust to None inputs and shape mismatches.

    Args:
        pred_mask: Binary mask from model prediction (HxW, dtype=bool or uint8). Can be None.
        gt_mask: Binary ground truth mask (HxW, dtype=bool or uint8). Can be None.

    Returns:
        IoU score (float), or 0.0 if inputs are invalid/mismatched.
    """
    if pred_mask is None or gt_mask is None:
        # print("Warning: calculate_miou received None mask.")
        return 0.0

    # Ensure boolean
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.shape != gt_mask.shape:
        print(f"Warning: Shape mismatch in calculate_miou: Pred {pred_mask.shape}, GT {gt_mask.shape}. Returning 0.")
        return 0.0 # Shape mismatch is a critical problem

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        # Both masks are empty. IoU is 1 if GT is also empty (implies intersection is 0), 0 otherwise.
        return 1.0 if intersection == 0 else 0.0

    iou = float(intersection) / float(union)
    return iou


def calculate_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """
    Calculate boundary F1 score using OpenCV Canny and Distance Transform.
    Robust to None inputs.

    Args:
        pred_mask: Binary prediction mask (HxW, dtype=bool or uint8). Can be None.
        gt_mask: Binary ground truth mask (HxW, dtype=bool or uint8). Can be None.
        tolerance_px: Pixel tolerance for boundary matching (default: 2).

    Returns:
        Boundary F1 score (float), or 0.0 if inputs are invalid.
    """
    if pred_mask is None or gt_mask is None:
        # print("Warning: calculate_boundary_f1 received None mask.")
        return 0.0

    # Ensure uint8 for OpenCV functions
    pred_mask_u8 = pred_mask.astype(np.uint8) * 255
    gt_mask_u8 = gt_mask.astype(np.uint8) * 255

    # Detect boundaries using Canny edge detection
    pred_boundary = cv2.Canny(pred_mask_u8, 100, 200) # Thresholds might need tuning
    gt_boundary = cv2.Canny(gt_mask_u8, 100, 200)

    # Count boundary pixels
    pred_sum = np.count_nonzero(pred_boundary)
    gt_sum = np.count_nonzero(gt_boundary)

    if gt_sum == 0 and pred_sum == 0:
        return 1.0  # Both empty, perfect match
    if gt_sum == 0 or pred_sum == 0:
        return 0.0  # One empty, the other not, zero score

    # Create distance transforms (distance to the nearest boundary point)
    # Invert boundary map so distance is 0 on the boundary
    gt_dist = cv2.distanceTransform(cv2.bitwise_not(gt_boundary), cv2.DIST_L2, 3)
    pred_dist = cv2.distanceTransform(cv2.bitwise_not(pred_boundary), cv2.DIST_L2, 3)

    # Calculate precision: Fraction of predicted boundary points close to GT boundary
    pred_boundary_points = pred_boundary > 0
    precision = np.sum(gt_dist[pred_boundary_points] <= tolerance_px) / pred_sum

    # Calculate recall: Fraction of GT boundary points close to predicted boundary
    gt_boundary_points = gt_boundary > 0
    recall = np.sum(pred_dist[gt_boundary_points] <= tolerance_px) / gt_sum

    # F1 score
    if precision + recall == 0:
        return 0.0

    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


# --- evaluate_robustness Updated for SAM2 Image Predictor --- #
def evaluate_robustness(sam2_image_predictor, test_dataset, degradation_fns, severity_levels, model_name="SAM2", prompt_type='point'):
    """
    Evaluates SAM2 model robustness using SAM2ImagePredictor across degradations.

    Args:
        sam2_image_predictor: The **loaded** SAM2ImagePredictor object.
        test_dataset: An iterable yielding (sample_id, image_rgb, gt_mask).
                      'image_rgb' should be HxWxC NumPy array (RGB format expected by SAM2).
                      'gt_mask' should be HxW NumPy array (binary/boolean).
        degradation_fns: Dict mapping degradation names (str) to functions.
                         Each function should take (image_rgb, severity) and return degraded_image_rgb.
        severity_levels: Dict mapping degradation names (str) to lists of severity values.
        model_name (str): Name of the model for logging purposes.
        prompt_type (str): Type of prompt ('point', 'box', 'auto'). Default 'point'.
                           NOTE: Only 'point' using GT center is implemented here.

    Returns:
        pandas.DataFrame with evaluation metrics for each image, degradation, and severity.
    """

    results = []
    # Predictor is passed in directly
    predictor = sam2_image_predictor

    total_samples = len(test_dataset) if hasattr(test_dataset, '__len__') else 'unknown'
    print(f"Starting SAM2 evaluation for {model_name} using Image Predictor on {total_samples} samples...")

    for i, (sample_id, image_rgb, gt_mask) in enumerate(test_dataset):
        # print(f"Processing sample {i+1}/{total_samples} (ID: {sample_id})...") # Verbose

        # --- Generate Default Prompt (Point at GT center) --- #
        # (Keep this logic, it's independent of predictor type)
        point_coords = None
        point_labels = None
        prompt_valid = False
        if prompt_type == 'point':
            gt_mask_uint8 = gt_mask.astype(np.uint8)
            if np.any(gt_mask_uint8): # Check if mask is not empty
                moments = cv2.moments(gt_mask_uint8)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    point_coords = np.array([[center_x, center_y]])
                    point_labels = np.array([1]) # 1 for foreground point
                    prompt_valid = True
                else: # Handle cases where moments are zero (e.g., single point mask)
                    y_indices, x_indices = np.where(gt_mask_uint8 > 0)
                    if len(x_indices) > 0:
                        center_x = int(np.mean(x_indices))
                        center_y = int(np.mean(y_indices))
                        point_coords = np.array([[center_x, center_y]])
                        point_labels = np.array([1])
                        prompt_valid = True

            if not prompt_valid:
                print(f" Warning: Sample {sample_id}: Could not generate valid point prompt (empty/invalid GT mask?). Recording NaN.")
                # Add NaN results for this sample and continue
                results.append({'model_name': model_name, 'sample_id': sample_id, 'degradation': 'none', 'severity': 0, 'miou': np.nan, 'boundary_f1': np.nan, 'score': np.nan})
                for deg_name in degradation_fns:
                     if deg_name in severity_levels:
                         for severity in severity_levels[deg_name]:
                             results.append({'model_name': model_name, 'sample_id': sample_id, 'degradation': deg_name, 'severity': severity, 'miou': np.nan, 'boundary_f1': np.nan, 'score': np.nan})
                continue # Skip to next sample

        # --- Helper function for SAM2 prediction using Image Predictor --- #
        @torch.no_grad() # Disable gradient calculations for inference
        def get_sam2_prediction(img_to_predict_rgb):
            """Gets SAM2 prediction and score for a single image and prompt using Image Predictor."""
            try:
                predictor.set_image(img_to_predict_rgb) # Expects RGB numpy array
                if prompt_type == 'point':
                    # Predict uses the image set previously
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False # Get single best mask
                    )
                    if masks is not None and len(masks) > 0 and scores is not None and len(scores) > 0:
                        # Output 'masks' is numpy array for ImagePredictor
                        pred_mask_np = masks[0] # [0] index might still be needed if predict returns list-like
                        pred_score = scores[0]
                        return pred_mask_np.astype(bool), pred_score # Return mask and score
                    else:
                        print(f"  Warning: Sample {sample_id}: SAM2 returned no mask/score for prompt.")
                        return None, None # Return None for mask and score
                # TODO: Add elif for 'box' or 'auto' prompts here if needed
                else:
                    print(f"Warning: Prompt type '{prompt_type}' not implemented in get_sam2_prediction.")
                    return None, None # Fallback for unimplemented types
            except Exception as e:
                print(f"  --- ERROR during SAM2 prediction for sample {sample_id}: {e} ---")
                # Consider more specific error handling if needed
                # import traceback
                # traceback.print_exc()
                return None, None # Return None on prediction error

        # --- 1. Baseline on clean image --- #
        clean_mask_pred, clean_score = get_sam2_prediction(image_rgb) # Get mask and score
        baseline_miou = calculate_miou(clean_mask_pred, gt_mask)
        baseline_bf1 = calculate_boundary_f1(clean_mask_pred, gt_mask)
        results.append({
            'model_name': model_name, 'sample_id': sample_id, 'degradation': 'none',
            'severity': 0, 'miou': baseline_miou, 'boundary_f1': baseline_bf1,
            'score': clean_score if clean_mask_pred is not None else np.nan # Store score, handle None mask case
        })

        # --- 2. Evaluate each degradation --- #
        for deg_name, deg_fn in degradation_fns.items():
            if deg_name not in severity_levels:
                print(f" Warning: No severity levels defined for degradation '{deg_name}'. Skipping.")
                continue

            # print(f"  Applying degradation: {deg_name}") # Verbose
            for severity in severity_levels[deg_name]:
                degraded_mask_pred = None
                degraded_score = np.nan # Default score to NaN
                miou = np.nan
                bf1 = np.nan
                try:
                    # Apply degradation (ensure it returns RGB)
                    degraded_img_rgb = deg_fn(image_rgb.copy(), severity)
                    # Get SAM2 prediction on degraded image
                    degraded_mask_pred, degraded_score = get_sam2_prediction(degraded_img_rgb) # Get mask and score
                    # Calculate metrics only if prediction succeeded
                    if degraded_mask_pred is not None:
                         miou = calculate_miou(degraded_mask_pred, gt_mask)
                         bf1 = calculate_boundary_f1(degraded_mask_pred, gt_mask)
                    # If prediction failed, miou/bf1 remain NaN, degraded_score remains NaN (or is None from predictor)

                except Exception as e:
                    print(f"  --- ERROR processing {deg_name} (severity {severity}) for sample {sample_id}: {e} ---")
                    # miou/bf1/score remain NaN

                results.append({
                    'model_name': model_name, 'sample_id': sample_id, 'degradation': deg_name,
                    'severity': severity, 'miou': miou, 'boundary_f1': bf1, 'score': degraded_score # Store score
                })

    print(f"\nEvaluation for {model_name} complete.")
    return pd.DataFrame(results)


# Example Usage (Illustrative - requires actual data, models, and VALID paths)
if __name__ == '__main__':
    print("--- Running metrics.py Main Example (Using Image Predictor) ---")
    print("This example requires:")
    print("  1. SAM2 installed via 'pip install -e ./external/sam2'")
    print("  2. Correct paths to SAM2 checkpoint (.pt) and config (.yaml) files.")
    print("  3. A CUDA-enabled GPU (or change DEVICE to 'cpu').")
    print("-" * 35)

    # --- IMPORTANT: Configuration - EDIT THESE PATHS ---
    # Find these files within the ./external/sam2 directory after cloning/downloading
    SAM2_CHECKPOINT_PATH = "./external/sam2/models/sam2_hiera_large.pt" # Example path - VERIFY
    SAM2_CONFIG_PATH = "./external/sam2/configs/sam2_hiera_l.yaml"    # Example path - VERIFY

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Dummy Degradation Functions (Keep or replace with yours) --- #
    def dummy_blur(image_rgb, severity):
        ksize = int(severity * 2) + 1
        ksize = ksize if ksize % 2 != 0 else ksize + 1 # Kernel must be odd
        # Ensure image is uint8 for GaussianBlur
        img_blur = cv2.GaussianBlur(image_rgb.astype(np.uint8), (ksize, ksize), 0)
        return img_blur.astype(image_rgb.dtype) # Cast back to original dtype if needed

    def dummy_noise(image_rgb, severity):
        # Generate noise scaled by severity, add, and clip
        noise = np.random.normal(0, severity, image_rgb.shape) # Severity as std dev
        noisy_image = np.clip(image_rgb.astype(np.float32) + noise, 0, 255)
        return noisy_image.astype(image_rgb.dtype) # Cast back

    # --- Create Dummy Dataset (Small RGB images) --- #
    dummy_dataset = []
    for i in range(2): # Number of sample images
        img_h, img_w = 64, 80 # Smaller dummy images
        # Create RGB image (uint8)
        img_rgb = (np.random.rand(img_h, img_w, 3) * 255).astype(np.uint8)
        # Create a simple ground truth mask (e.g., a circle)
        mask = np.zeros((img_h, img_w), dtype=bool)
        center_x, center_y = img_w // 2, img_h // 2
        radius = min(img_h, img_w) // 4
        rr, cc = np.ogrid[:img_h, :img_w]
        mask_area = (rr - center_y)**2 + (cc - center_x)**2 <= radius**2
        mask[mask_area] = True
        dummy_dataset.append((f"dummy_{i}", img_rgb, mask))

    # --- Define Degradations and Levels --- #
    degradation_functions = { 'blur': dummy_blur, 'noise': dummy_noise }
    severity_levels = { 'blur': [1, 5, 10], 'noise': [10, 25, 50] } # Example severities

    # --- Load Model and Run Evaluation --- #
    sam_predictor = None # Use Image Predictor
    try:
        # Attempt to load the SAM2 image predictor
        sam_predictor = load_sam2_image_predictor(SAM2_CHECKPOINT_PATH, SAM2_CONFIG_PATH, device=DEVICE)

    except FileNotFoundError:
        print("\n---! ERROR: SAM2 checkpoint or config path not found !---")
        print(f"  Checkpoint tried: {SAM2_CHECKPOINT_PATH}")
        print(f"  Config tried: {SAM2_CONFIG_PATH}")
        print("  Please VERIFY these paths in the script and ensure the files exist.")
    except ImportError:
         print("\n---! ERROR: Failed to import SAM2 components !---")
         print("  Ensure SAM2 is installed from the 'external/sam2' directory.")
    except Exception as e:
        print(f"\n---! An unexpected error occurred during model loading: {e} !---")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed stack trace

    # Proceed only if predictor loaded successfully
    if sam_predictor:
        print("\n--- Running Evaluation ---")
        try:
            results_df = evaluate_robustness(
                sam2_image_predictor=sam_predictor, # Pass the predictor
                test_dataset=dummy_dataset,
                degradation_fns=degradation_functions,
                severity_levels=severity_levels,
                model_name="SAM2_DummyEval"
            )
            print("\n--- Evaluation Results DataFrame (Raw) ---")
            # Display options for potentially wide tables
            pd.set_option('display.max_rows', 50)
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.width', 120)
            print(results_df)

            # +++ Add Post-Processing for Robustness Metrics +++
            if not results_df.empty and 'miou' in results_df.columns: # Check if df is valid
                print("\n--- Calculating Robustness Metrics ---")

                # 1. Extract Baseline Scores
                baseline_scores = results_df[results_df['degradation'] == 'none'][['sample_id', 'miou', 'boundary_f1', 'score']].rename(
                    columns={'miou': 'baseline_miou', 'boundary_f1': 'baseline_bf1', 'score': 'baseline_score'}
                ).set_index('sample_id') # Set index for easy merging/lookup

                # 2. Isolate Degraded Results
                degraded_results = results_df[results_df['degradation'] != 'none'].copy()

                # Check if there are any degraded results to process
                if not degraded_results.empty:
                    # 3. Merge Baseline Scores onto Degraded Results
                    # Using join is efficient when baseline_scores index is set to sample_id
                    results_with_baseline = degraded_results.join(baseline_scores, on='sample_id')

                    # 4. Calculate Robustness Metrics (handle NaNs and division by zero)
                    # Ratios (Higher is better, 1.0 = perfect maintenance)
                    results_with_baseline['miou_ratio'] = np.where(
                        (results_with_baseline['baseline_miou'].notna()) & (results_with_baseline['baseline_miou'] != 0) & (results_with_baseline['miou'].notna()),
                        results_with_baseline['miou'] / results_with_baseline['baseline_miou'],
                        np.nan # Assign NaN if baseline is NaN/zero or degraded is NaN
                    )
                    results_with_baseline['bf1_ratio'] = np.where(
                        (results_with_baseline['baseline_bf1'].notna()) & (results_with_baseline['baseline_bf1'] != 0) & (results_with_baseline['boundary_f1'].notna()),
                        results_with_baseline['boundary_f1'] / results_with_baseline['baseline_bf1'],
                        np.nan
                    )
                    results_with_baseline['score_ratio'] = np.where(
                         (results_with_baseline['baseline_score'].notna()) & (results_with_baseline['baseline_score'] != 0) & (results_with_baseline['score'].notna()),
                         results_with_baseline['score'] / results_with_baseline['baseline_score'],
                         np.nan
                     )

                    # Absolute Drops (Lower is better, 0 = perfect maintenance)
                    results_with_baseline['miou_drop_abs'] = np.where(
                         (results_with_baseline['baseline_miou'].notna()) & (results_with_baseline['miou'].notna()),
                         results_with_baseline['baseline_miou'] - results_with_baseline['miou'],
                         np.nan
                    )
                    results_with_baseline['bf1_drop_abs'] = np.where(
                         (results_with_baseline['baseline_bf1'].notna()) & (results_with_baseline['boundary_f1'].notna()),
                         results_with_baseline['baseline_bf1'] - results_with_baseline['boundary_f1'],
                         np.nan
                    )
                    results_with_baseline['score_drop_abs'] = np.where(
                         (results_with_baseline['baseline_score'].notna()) & (results_with_baseline['score'].notna()),
                         results_with_baseline['baseline_score'] - results_with_baseline['score'],
                         np.nan
                     )

                    print("\n--- Robustness Metrics DataFrame (per sample/degradation/severity) ---")
                    # Select and display key columns including the new robustness metrics
                    display_cols = ['sample_id', 'degradation', 'severity',
                                    'miou', 'baseline_miou', 'miou_ratio', 'miou_drop_abs',
                                    'boundary_f1', 'baseline_bf1', 'bf1_ratio', 'bf1_drop_abs',
                                    'score', 'baseline_score', 'score_ratio', 'score_drop_abs']
                    # Filter out columns not present in the df just in case
                    display_cols = [col for col in display_cols if col in results_with_baseline.columns]
                    print(results_with_baseline[display_cols].round(4)) # Round for readability

                    # 5. Calculate and Print Average Robustness Ratios
                    print("\n--- Average Robustness Ratios (relative to baseline, higher is better) ---")
                    # Calculate mean ratio, ignoring NaNs. Group by degradation and severity.
                    avg_robustness_ratios = results_with_baseline.groupby(['degradation', 'severity'])[['miou_ratio', 'bf1_ratio', 'score_ratio']].mean().round(4)
                    print(avg_robustness_ratios)

                    # Optional: Print Average Absolute Drops
                    # print("\n--- Average Absolute Performance Drop (lower is better) ---")
                    # avg_robustness_drops = results_with_baseline.groupby(['degradation', 'severity'])[['miou_drop_abs', 'bf1_drop_abs', 'score_drop_abs']].mean().round(4)
                    # print(avg_robustness_drops)

                else:
                    print("\nSkipping robustness metric calculation (no degraded results found).")

            elif results_df.empty:
                print("\nRobustness analysis skipped (no results generated).")
            else:
                print("\nRobustness analysis skipped (results DataFrame missing expected columns).")

            # --- Original Degradation Tolerance Example (Optional, keep or remove) ---
            # This part calculates the max severity passed based on a threshold (e.g., 50% of baseline mIoU)
            # It's a different way to summarize robustness compared to ratios/drops.
            print("\n--- Degradation Tolerance Example (Max Severity with mIoU >= 50% of Baseline) ---")
            # Check if baseline_iou_map can be created
            if not results_df[results_df['degradation'] == 'none'].empty:
                baseline_iou_map = results_df[results_df['degradation'] == 'none'].set_index('sample_id')['miou']
                tolerance_results = {}
                if not baseline_iou_map.empty:
                    for sample_id, baseline_iou in baseline_iou_map.items():
                        if pd.isna(baseline_iou) or baseline_iou <= 0: # Handle failed/zero/negative baseline
                            tolerance_results[sample_id] = {deg_name: 'N/A (Bad Baseline)' for deg_name in degradation_functions.keys()}
                            continue

                        sample_data = results_df[results_df['sample_id'] == sample_id]
                        tolerance_results[sample_id] = {}
                        threshold = baseline_iou * 0.5 # Example: 50% threshold
                        for deg_name in degradation_functions.keys():
                            # Ensure the degradation exists in the results for this sample
                            degraded_data = sample_data[(sample_data['degradation'] == deg_name) & sample_data['miou'].notna()].sort_values('severity')
                            if degraded_data.empty:
                                 tolerance_results[sample_id][deg_name] = 'N/A (No Data)'
                                 continue

                            passed_levels = degraded_data[degraded_data['miou'] >= threshold]['severity']
                            if len(passed_levels) == 0:
                                tolerance_results[sample_id][deg_name] = '< Min Level Tested' # Failed even at lowest severity
                            else:
                                # Report the highest severity level passed
                                tolerance_results[sample_id][deg_name] = passed_levels.max()
                        else: # Handle cases where no degradation_functions exist? Or only print if tolerance_results has data?
                           if tolerance_results: # Only print if there's something to show
                                print(pd.DataFrame.from_dict(tolerance_results, orient='index'))

                else:
                     print("\nDegradation tolerance analysis skipped (no baseline results to create map).")
            else:
                 print("\nDegradation tolerance analysis skipped (no baseline results found in DataFrame).")
            # --- End Tolerance Example ---

        except Exception as e:
            print(f"\n---! An error occurred during evaluation run or post-processing: {e} !---")
            import traceback
            traceback.print_exc() # Print stack trace for debugging main block errors

    else:
        print("\n--- Evaluation skipped because the SAM2 predictor could not be loaded. ---")

    print("\n--- metrics.py Main Example Finished ---")