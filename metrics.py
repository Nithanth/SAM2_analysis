import numpy as np
import cv2
import pandas as pd

def calculate_miou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between a predicted binary mask 
    and a ground truth binary mask.

    Args:
        pred_mask: Binary mask from model prediction (HxW, dtype=bool or uint8).
        gt_mask: Binary ground truth mask (HxW, dtype=bool or uint8).

    Returns:
        IoU score (float).
    """
    # Ensure masks are boolean
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        # Handle case where both masks are empty
        # If prediction is also empty, IoU is 1, otherwise 0
        return 1.0 if intersection == 0 else 0.0 
    
    iou = intersection / union
    return iou

def calculate_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """
    Calculate boundary F1 score for segmentation accuracy at the edges.

    Args:
        pred_mask: Binary prediction mask (HxW, dtype=bool or uint8).
        gt_mask: Binary ground truth mask (HxW, dtype=bool or uint8).
        tolerance_px: Pixel tolerance for boundary matching (default: 2).

    Returns:
        Boundary F1 score (float).
    """
    # Ensure masks are uint8 for OpenCV functions
    pred_mask_u8 = pred_mask.astype(np.uint8) * 255
    gt_mask_u8 = gt_mask.astype(np.uint8) * 255

    # Extract boundaries using Canny edge detection
    pred_boundary = cv2.Canny(pred_mask_u8, 100, 200)
    gt_boundary = cv2.Canny(gt_mask_u8, 100, 200)

    # Check if either boundary is empty
    if np.sum(gt_boundary) == 0 and np.sum(pred_boundary) == 0:
        return 1.0  # Both empty, perfect match
    if np.sum(gt_boundary) == 0 or np.sum(pred_boundary) == 0:
        return 0.0  # One empty, the other not, zero score

    # Create distance transforms (distance to the nearest boundary point)
    # Invert boundary maps so distance transform finds distance to edges
    gt_dist = cv2.distanceTransform(cv2.bitwise_not(gt_boundary), cv2.DIST_L2, 3)
    pred_dist = cv2.distanceTransform(cv2.bitwise_not(pred_boundary), cv2.DIST_L2, 3)

    # Calculate precision: Fraction of predicted boundary points close to ground truth boundary
    pred_boundary_points = pred_boundary > 0
    precision = np.sum(gt_dist[pred_boundary_points] <= tolerance_px) / np.sum(pred_boundary_points)

    # Calculate recall: Fraction of ground truth boundary points close to predicted boundary
    gt_boundary_points = gt_boundary > 0
    recall = np.sum(pred_dist[gt_boundary_points] <= tolerance_px) / np.sum(gt_boundary_points)

    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def predict_mask(model, image, prompt=None):
    """ 
    Placeholder function for model inference. 
    Replace this with actual calls to SAM2 or baseline model.
    
    Args:
        model: The segmentation model object.
        image: Input image (e.g., NumPy array).
        prompt: Optional prompt for SAM2 (e.g., points, boxes).
        
    Returns:
        Predicted binary mask (NumPy array).
    """
    # --- Placeholder Implementation --- 
    # Replace this with your actual model inference logic
    print("Warning: Using placeholder predict_mask function.")
    # Example: return a dummy mask of zeros
    return np.zeros(image.shape[:2], dtype=bool) 
    # -------------------------------

def evaluate_robustness(model, test_dataset, degradation_fns, severity_levels, model_name="model"):
    """
    Evaluates model robustness across specified degradations and severities.

    Args:
        model: The segmentation model to evaluate.
        test_dataset: An iterable yielding (sample_id, image, gt_mask).
                      'image' should be HxWxC NumPy array (BGR or RGB).
                      'gt_mask' should be HxW NumPy array (binary/boolean).
        degradation_fns: Dict mapping degradation names (str) to functions.
                         Each function should take (image, severity) and return degraded_image.
        severity_levels: Dict mapping degradation names (str) to lists of severity values.
        model_name: Name of the model for logging purposes.

    Returns:
        pandas.DataFrame with evaluation metrics for each image, degradation, and severity.
    """
    results = []
    
    total_samples = len(test_dataset) if hasattr(test_dataset, '__len__') else 'unknown'
    print(f"Starting evaluation for {model_name} on {total_samples} samples...")

    for i, (sample_id, image, gt_mask) in enumerate(test_dataset):
        print(f"Processing sample {i+1}/{total_samples} (ID: {sample_id})...")
        
        # --- 1. Baseline on clean image --- 
        try:
            # Note: Adapt predict_mask call based on your model's needs (e.g., prompts for SAM2)
            clean_mask = predict_mask(model, image) 
            baseline_miou = calculate_miou(clean_mask, gt_mask)
            baseline_bf1 = calculate_boundary_f1(clean_mask, gt_mask)
            
            results.append({
                'model_name': model_name,
                'sample_id': sample_id,
                'degradation': 'none',
                'severity': 0,
                'miou': baseline_miou,
                'boundary_f1': baseline_bf1
            })
        except Exception as e:
            print(f" Error processing baseline for sample {sample_id}: {e}")
            results.append({
                'model_name': model_name,
                'sample_id': sample_id,
                'degradation': 'none',
                'severity': 0,
                'miou': np.nan, # Indicate error
                'boundary_f1': np.nan
            })

        # --- 2. Evaluate each degradation --- 
        for deg_name, deg_fn in degradation_fns.items():
            if deg_name not in severity_levels:
                print(f" Warning: No severity levels defined for degradation '{deg_name}'. Skipping.")
                continue
                
            print(f"  Applying degradation: {deg_name}")
            for severity in severity_levels[deg_name]:
                try:
                    degraded_img = deg_fn(image.copy(), severity) # Use copy to avoid modifying original
                    
                    # Note: Adapt predict_mask call based on your model's needs
                    degraded_mask = predict_mask(model, degraded_img)
                    
                    miou = calculate_miou(degraded_mask, gt_mask)
                    bf1 = calculate_boundary_f1(degraded_mask, gt_mask)
                    
                    results.append({
                        'model_name': model_name,
                        'sample_id': sample_id,
                        'degradation': deg_name,
                        'severity': severity,
                        'miou': miou,
                        'boundary_f1': bf1
                    })
                except Exception as e:
                    print(f"   Error processing {deg_name} (severity {severity}) for sample {sample_id}: {e}")
                    results.append({
                        'model_name': model_name,
                        'sample_id': sample_id,
                        'degradation': deg_name,
                        'severity': severity,
                        'miou': np.nan, # Indicate error
                        'boundary_f1': np.nan
                    })

    print(f"Evaluation for {model_name} complete.")
    return pd.DataFrame(results)

# Example Usage (Illustrative - requires actual data and models)
if __name__ == '__main__':
    # --- Dummy Data & Functions for Illustration --- 
    class DummyModel:
        def __call__(self, image):
            # Simulate prediction - replace with real model
            return (np.random.rand(*image.shape[:2]) > 0.5) # Random mask
    
    def dummy_blur(image, severity):
        ksize = int(severity * 2) + 1 # Example severity mapping
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def dummy_noise(image, severity):
        noise = np.random.normal(0, severity * 25, image.shape).astype(image.dtype)
        return cv2.add(image, noise)

    # Create dummy dataset (replace with your actual data loader)
    dummy_dataset = []
    for i in range(3): # 3 sample images
        img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        mask = (np.random.rand(100, 100) > 0.6) # Random ground truth
        dummy_dataset.append((f"img_{i}", img, mask))

    # Define degradations and levels
    degradation_functions = {
        'blur': dummy_blur,
        'noise': dummy_noise
    }
    severity_levels = {
        'blur': [1, 3, 5], # Example severity values
        'noise': [0.05, 0.1, 0.2] # Example severity values
    }
    
    # --- Run Evaluation --- 
    model = DummyModel()
    results_df = evaluate_robustness(
        model=model, 
        test_dataset=dummy_dataset, 
        degradation_fns=degradation_functions, 
        severity_levels=severity_levels,
        model_name="dummy_model"
    )
    
    print("\n--- Evaluation Results ---")
    print(results_df)

    # --- Post-Hoc Degradation Tolerance (Example) ---
    print("\n--- Degradation Tolerance Example (mIoU < 50% of baseline) ---")
    baseline_iou_map = results_df[results_df['degradation'] == 'none'].set_index('sample_id')['miou']
    tolerance_results = {}
    for sample_id, baseline_iou in baseline_iou_map.items():
        sample_data = results_df[results_df['sample_id'] == sample_id]
        tolerance_results[sample_id] = {}
        threshold = baseline_iou * 0.5
        for deg_name in degradation_functions.keys():
            degraded_data = sample_data[sample_data['degradation'] == deg_name].sort_values('severity')
            passed_levels = degraded_data[degraded_data['miou'] >= threshold]['severity']
            if len(passed_levels) == 0:
                 tolerance_results[sample_id][deg_name] = None # Failed at lowest level
            else:
                 tolerance_results[sample_id][deg_name] = passed_levels.max()

    print(pd.DataFrame.from_dict(tolerance_results, orient='index'))

