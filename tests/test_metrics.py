# Basic unit tests for metric calculations
import numpy as np
import pytest

from metrics import calculate_miou, calculate_boundary_f1


@pytest.fixture
def sample_masks():
    """Provides simple masks for testing."""
    mask_full = np.ones((10, 10), dtype=np.uint8)
    mask_empty = np.zeros((10, 10), dtype=np.uint8)
    mask_half = np.zeros((10, 10), dtype=np.uint8)
    mask_half[5:, :] = 1
    return mask_full, mask_empty, mask_half


def test_miou_perfect(sample_masks):
    mask_full, mask_empty, _ = sample_masks
    assert calculate_miou(mask_full, mask_full) == 1.0
    assert calculate_miou(mask_empty, mask_empty) == 1.0  # Empty union/intersection handled


def test_miou_none(sample_masks):
    mask_full, _, _ = sample_masks
    assert calculate_miou(None, mask_full) == 0.0
    assert calculate_miou(mask_full, None) == 0.0
    assert calculate_miou(None, None) == 0.0


def test_miou_half(sample_masks):
    mask_full, _, mask_half = sample_masks
    # Intersection is 50, Union is 100
    assert calculate_miou(mask_half, mask_full) == 0.5


def test_bf1_perfect(sample_masks):
    mask_full, mask_empty, _ = sample_masks
    assert calculate_boundary_f1(mask_full, mask_full, tolerance_px=1) == 1.0
    assert calculate_boundary_f1(mask_empty, mask_empty, tolerance_px=1) == 1.0


def test_bf1_none(sample_masks):
    mask_full, _, _ = sample_masks
    assert calculate_boundary_f1(None, mask_full) == 0.0
    assert calculate_boundary_f1(mask_full, None) == 0.0
    assert calculate_boundary_f1(None, None) == 0.0


# Add more tests for mismatch, different shapes (should return 0), etc. if needed

