import numpy as np
import cv2 

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
        # Both masks are empty. IoU is 1 if GT is also empty (implies intersection is 0), 0 otherwise
        return 1.0 if intersection == 0 else 0.0

    iou = float(intersection) / float(union)
    return iou


def calculate_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """
    Calculate boundary F1 score using OpenCV Canny and Distance Transform.
    Robust to None inputs and shape mismatches.

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

    # Added shape check for robustness
    if pred_mask.shape != gt_mask.shape:
        print(f"Warning: Shape mismatch in calculate_boundary_f1: Pred {pred_mask.shape}, GT {gt_mask.shape}. Returning 0.")
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
    gt_dist_map = cv2.distanceTransform(cv2.bitwise_not(gt_boundary), cv2.DIST_L2, 3)
    pred_boundary_pixels = pred_boundary > 0
    pred_matched_count = np.sum(gt_dist_map[pred_boundary_pixels] <= tolerance_px)

    # Calculate distance from Pred boundary to GT boundary pixels
    pred_dist_map = cv2.distanceTransform(1 - (pred_boundary // 255), cv2.DIST_L2, 3)
    gt_boundary_pixels = np.where(gt_boundary == 255)
    gt_matched_count = np.sum(pred_dist_map[gt_boundary_pixels] <= tolerance_px)

    # Calculate Precision, Recall, and F1 Score
    precision = float(pred_matched_count) / float(pred_sum) if pred_sum > 0 else 0
    recall = float(gt_matched_count) / float(gt_sum) if gt_sum > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1_score = 2.0 * (precision * recall) / (precision + recall)

    return f1_score


def _create_test_circle_mask(shape: tuple, center: tuple, radius: int) -> np.ndarray:
    """Helper to create a binary mask with a filled circle."""
    mask = np.zeros(shape, dtype=np.uint8)
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask[dist_from_center <= radius] = 1
    return mask

if __name__ == "__main__":
    # --- Test Cases for Metric Functions --- #
    print("Running test cases for metric functions...")
    tolerance = 1e-6 # Tolerance for floating point comparisons
    shape = (50, 50) # Use a larger, more realistic shape for geometry tests
    radius = 10

    # --- Test Case 1: Perfect Overlap (Circles) --- #
    center1 = (shape[1]//2, shape[0]//2) # Center of the image
    mask_perfect1 = _create_test_circle_mask(shape, center1, radius)
    mask_perfect2 = _create_test_circle_mask(shape, center1, radius)
    assert abs(calculate_miou(mask_perfect1, mask_perfect2) - 1.0) < tolerance, "Test Case 1 Failed (mIoU - Circle)"
    # BF1 might not be exactly 1.0 due to pixel discretization of boundary, check it's very close
    assert abs(calculate_boundary_f1(mask_perfect1, mask_perfect2) - 1.0) < 1e-2, "Test Case 1 Failed (BF1 - Circle)"
    print("Test Case 1 (Perfect Overlap - Circle): Passed")

    # --- Test Case 2: No Overlap (Circles) --- #
    center_no_overlap1 = (shape[1]//4, shape[0]//4) # Top-left quadrant
    center_no_overlap2 = (3*shape[1]//4, 3*shape[0]//4) # Bottom-right quadrant
    mask_no_overlap1 = _create_test_circle_mask(shape, center_no_overlap1, radius)
    mask_no_overlap2 = _create_test_circle_mask(shape, center_no_overlap2, radius)
    assert abs(calculate_miou(mask_no_overlap1, mask_no_overlap2) - 0.0) < tolerance, "Test Case 2 Failed (mIoU - Circle)"
    # BF1 should be exactly 0.0 as boundaries are far apart
    assert abs(calculate_boundary_f1(mask_no_overlap1, mask_no_overlap2) - 0.0) < tolerance, "Test Case 2 Failed (BF1 - Circle)"
    print("Test Case 2 (No Overlap - Circle): Passed")

    # --- Test Case 3: Partial Overlap (Circles) --- #
    center_partial1 = (shape[1]//2 - radius//2, shape[0]//2) # Slightly left of center
    center_partial2 = (shape[1]//2 + radius//2, shape[0]//2) # Slightly right of center
    mask_partial1 = _create_test_circle_mask(shape, center_partial1, radius)
    mask_partial2 = _create_test_circle_mask(shape, center_partial2, radius)
    # Calculate expected IoU numerically for this specific overlap
    intersection_area = np.logical_and(mask_partial1, mask_partial2).sum()
    union_area = np.logical_or(mask_partial1, mask_partial2).sum()
    expected_iou_partial = float(intersection_area) / float(union_area) if union_area > 0 else 0
    print(f"  [INFO TC3] Calculated Expected Partial IoU (Circles): {expected_iou_partial:.4f}")
    assert abs(calculate_miou(mask_partial1, mask_partial2) - expected_iou_partial) < tolerance, "Test Case 3 Failed (mIoU - Circle)"
    bf1_partial = calculate_boundary_f1(mask_partial1, mask_partial2)
    assert 0.0 < bf1_partial < 1.0, f"Test Case 3 Failed (BF1 bounds - Circle, got {bf1_partial})"
    print(f"Test Case 3 (Partial Overlap - Circle): Passed (mIoU={expected_iou_partial:.4f}, BF1={bf1_partial:.4f})")

    # --- Test Case 4: Both Masks Empty --- #
    mask_empty1 = np.zeros((5, 5), dtype=np.uint8) # Keep small for this specific test
    mask_empty2 = np.zeros((5, 5), dtype=np.uint8)
    assert abs(calculate_miou(mask_empty1, mask_empty2) - 1.0) < tolerance, "Test Case 4 Failed (mIoU - Both Empty)"
    assert abs(calculate_boundary_f1(mask_empty1, mask_empty2) - 1.0) < tolerance, "Test Case 4 Failed (BF1 - Both Empty)"
    print("Test Case 4 (Both Empty): Passed")

    # --- Test Case 5: One Mask Empty --- #
    mask_one_empty1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    mask_one_empty2 = np.zeros((2, 2), dtype=np.uint8)
    assert abs(calculate_miou(mask_one_empty1, mask_one_empty2) - 0.0) < tolerance, "Test Case 5 Failed (mIoU - One Empty)"
    assert abs(calculate_boundary_f1(mask_one_empty1, mask_one_empty2) - 0.0) < tolerance, "Test Case 5 Failed (BF1 - One Empty)"
    # Test the other way around too
    assert abs(calculate_miou(mask_one_empty2, mask_one_empty1) - 0.0) < tolerance, "Test Case 5 Failed (mIoU - One Empty Reversed)"
    assert abs(calculate_boundary_f1(mask_one_empty2, mask_one_empty1) - 0.0) < tolerance, "Test Case 5 Failed (BF1 - One Empty Reversed)"
    print("Test Case 5 (One Empty): Passed")

    # --- Test Case 6: None Inputs --- #
    mask_valid = np.ones((3, 3), dtype=np.uint8)
    assert abs(calculate_miou(None, mask_valid) - 0.0) < tolerance, "Test Case 6 Failed (mIoU - None Input 1)"
    assert abs(calculate_miou(mask_valid, None) - 0.0) < tolerance, "Test Case 6 Failed (mIoU - None Input 2)"
    assert abs(calculate_miou(None, None) - 0.0) < tolerance, "Test Case 6 Failed (mIoU - None Input 3)"
    assert abs(calculate_boundary_f1(None, mask_valid) - 0.0) < tolerance, "Test Case 6 Failed (BF1 - None Input 1)"
    assert abs(calculate_boundary_f1(mask_valid, None) - 0.0) < tolerance, "Test Case 6 Failed (BF1 - None Input 2)"
    assert abs(calculate_boundary_f1(None, None) - 0.0) < tolerance, "Test Case 6 Failed (BF1 - None Input 3)"
    print("Test Case 6 (None Inputs): Passed")

    # --- Test Case 7: Shape Mismatch (Should print warning and return 0) --- #
    mask_shape1 = np.ones((2, 2), dtype=np.uint8)
    mask_shape2 = np.ones((3, 3), dtype=np.uint8)
    print("\nTesting Shape Mismatch (expect a warning message):")
    assert abs(calculate_miou(mask_shape1, mask_shape2) - 0.0) < tolerance, "Test Case 7 Failed (mIoU - Shape Mismatch)"
    # With the added shape check, BF1 should now return 0.0 directly
    bf1_shape_mismatch = calculate_boundary_f1(mask_shape1, mask_shape2)
    assert abs(bf1_shape_mismatch - 0.0) < tolerance, "Test Case 7 Failed (BF1 - Shape Mismatch, expected 0)"
    print("Test Case 7 (Shape Mismatch): Passed (check warning/output)")

    print("\nAll metric test cases completed.")
