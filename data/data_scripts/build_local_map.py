"""Generate 'degradation_map.json' from local image/annotation structure.

Scans the following directories relative to the project root:
- data/images/gt_img/        (for <id>.jpg and <id>_annotations.json)
- data/images/img_degraded/  (for <degradation_type>/<id>_*_<level>.jpg)

Produces 'data/degradation_map.json' with the structure expected by
`sam2_eval_pipeline.py`.

Run directly from the project root directory:
    python data/data_scripts/build_local_map.py

Requirements:
    pip install pycocotools Pillow
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from pycocotools import mask as mask_utils

# -----------------------------------------------------------------------------
# Configuration (Adjust if your structure differs)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # Project root
DATA_DIR = ROOT_DIR / "data"
GT_IMG_DIR = DATA_DIR / "images" / "gt_img"         # Originals + Annotations
DEG_IMG_DIR = DATA_DIR / "images" / "img_degraded" # Degraded images sub-folders
OUTPUT_JSON = DATA_DIR / "degradation_map.json"   # Output file location

# Regex to extract ID and Level from degraded filenames (adjust if needed)
# Assumes format like: <id>_<typename>_<level>.<ext>
# Example: 1_gaussian_blur_5.jpg -> id=1, level=5
# Example: 2_jpeg_70.jpg -> id=2, level=70
# Captures: 1: id, 2: level (numeric part)
FILENAME_PATTERN = re.compile(r"^(\d+)_.*?_(\d+(?:\.\d+)?)\.(?:jpg|png|jpeg)$", re.IGNORECASE)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_first_polygon(anno_path: Path) -> List[float] | None:
    """Loads annotation JSON and returns the first polygon segmentation found."""
    if not anno_path.is_file():
        print(f"Warning: Annotation file not found: {anno_path}")
        return None
    try:
        with open(anno_path, "r") as f:
            data = json.load(f)

        # Normalize to list[annotation]
        anns = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "annotations" in data:
            anns = data["annotations"]
        if not isinstance(anns, list):
            anns = [anns] # Handle case where it's a single dict not in a list

        # Find the first annotation with a polygon segmentation
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
            if isinstance(seg, dict):  # Skip if it's already RLE
                print(f"Warning: Annotation {anno_path.name} contains RLE, expected polygon. Skipping polygon search.")
                return None
            if isinstance(seg, list):
                 # COCO polygon format: list[list[float]] or list[float]
                 # Take the first polygon list
                polygon = seg[0] if seg and isinstance(seg[0], list) else seg
                if polygon and isinstance(polygon[0], (int, float)):
                     # Basic check for coordinate format
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:
                         return polygon
                    else:
                        print(f"Warning: Invalid polygon format in {anno_path.name}: {polygon}")
                else:
                    print(f"Warning: Unexpected polygon structure in {anno_path.name}: {seg}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {anno_path}")
    except Exception as e:
        print(f"Error processing annotation {anno_path}: {e}")
    return None

def polygon_to_rle(polygon: List[float], height: int, width: int) -> Dict | None:
    """Converts a COCO polygon to RLE format."""
    if not polygon:
        return None
    try:
        rle_list = mask_utils.frPyObjects([polygon], height, width)
        # Merge RLEs if it returns a list (should be single polygon here)
        rle = mask_utils.merge(rle_list) if isinstance(rle_list, list) else rle_list
        # Ensure counts is string for JSON serialisation
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        # Ensure size is [height, width]
        rle["size"] = [height, width]
        return rle
    except Exception as e:
        print(f"Error converting polygon to RLE: {e}")
        return None

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def build_degradation_map() -> Dict[str, Dict]:
    """Scans directories and builds the degradation map."""
    degradation_map = {}
    print(f"Scanning ground truth images and annotations in: {GT_IMG_DIR}")

    processed_ids = set()

    # Iterate through potential annotation files first
    for anno_file in GT_IMG_DIR.glob("*_annotations.json"):
        if not anno_file.is_file():
            continue

        img_id_match = re.match(r"^(\d+)_annotations\.json$", anno_file.name)
        if not img_id_match:
            print(f"Warning: Skipping file with unexpected name format: {anno_file.name}")
            continue

        img_id = img_id_match.group(1)
        if img_id in processed_ids:
             print(f"Warning: Duplicate annotation file found for ID {img_id}, skipping {anno_file.name}")
             continue

        # --- Find corresponding image --- 
        img_file = GT_IMG_DIR / f"{img_id}.jpg"
        if not img_file.is_file():
            # Try other extensions if needed
            img_file_png = GT_IMG_DIR / f"{img_id}.png"
            if img_file_png.is_file():
                img_file = img_file_png
            else:
                 print(f"Warning: Original image for annotation {anno_file.name} (ID: {img_id}) not found. Searched for {img_id}.jpg/.png. Skipping this ID.")
                 continue

        print(f"Processing ID: {img_id} (Image: {img_file.name}, Annotation: {anno_file.name})")

        # --- Get Image Dimensions --- 
        try:
            with Image.open(img_file) as im:
                width, height = im.size # PIL gives W, H
        except Exception as e:
            print(f"Error opening image {img_file}: {e}. Skipping ID {img_id}.")
            continue

        # --- Get Ground Truth Polygon and Convert to RLE --- 
        polygon = get_first_polygon(anno_file)
        if not polygon:
            print(f"Warning: Could not get valid polygon from {anno_file.name}. Skipping ID {img_id}.")
            continue

        gt_rle = polygon_to_rle(polygon, height, width) # Note: H, W order for RLE size
        if not gt_rle:
            print(f"Warning: Failed to convert polygon to RLE for ID {img_id}. Skipping.")
            continue

        # --- Initialize entry for this image_id --- 
        degradation_map[img_id] = {
            "ground_truth_rle": gt_rle,
            "versions": {},
        }

        # --- Add Original Version --- 
        # Path relative to DATA_DIR (which is likely the image_base_dir in config)
        original_rel_path = Path("images") / "gt_img" / img_file.name
        degradation_map[img_id]["versions"]["original"] = {
            "filepath": str(original_rel_path),
            "level": 0,
            "degradation_type": "original",
        }

        # --- Scan for Degraded Versions --- 
        if not DEG_IMG_DIR.is_dir():
             print(f"Warning: Degraded images directory not found: {DEG_IMG_DIR}")
             continue # Move to next image ID if no degraded dir exists

        for deg_type_dir in DEG_IMG_DIR.iterdir():
            if not deg_type_dir.is_dir():
                continue
            deg_type = deg_type_dir.name

            # Add dict for this degradation type if not present
            if deg_type not in degradation_map[img_id]["versions"]:
                degradation_map[img_id]["versions"][deg_type] = {}

            for deg_file in deg_type_dir.glob(f"{img_id}_*.jpg"):
                 match = FILENAME_PATTERN.match(deg_file.name)
                 if match and match.group(1) == img_id:
                     level_str = match.group(2)
                     try:
                         # Attempt to convert level to float or int
                         level_val = float(level_str) if '.' in level_str else int(level_str)
                     except ValueError:
                          print(f"Warning: Could not parse level '{level_str}' from filename {deg_file.name}. Skipping this file.")
                          continue

                     # Path relative to DATA_DIR
                     deg_rel_path = Path("images") / "img_degraded" / deg_type / deg_file.name
                     degradation_map[img_id]["versions"][deg_type][level_str] = {
                         "filepath": str(deg_rel_path),
                         "level": level_val,
                         "degradation_type": deg_type,
                     }
                 # Add .png or other extensions if needed
                 # elif deg_file.suffix.lower() == '.png': ...

        processed_ids.add(img_id)

    if not degradation_map:
        print("Error: No valid image/annotation pairs found. Please check directories:")
        print(f"- Ground Truth/Annotations: {GT_IMG_DIR}")
        print(f"- Degraded Images: {DEG_IMG_DIR}")

    return degradation_map


def main() -> None:
    """Script entry point: build map and save to JSON."""
    print("Starting degradation map generation...")
    degradation_map = build_degradation_map()

    if degradation_map:
        # Ensure the output directory exists (it should, it's just DATA_DIR)
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving degradation map to: {OUTPUT_JSON}")
        try:
            with open(OUTPUT_JSON, "w") as fo:
                json.dump(degradation_map, fo, indent=2)
            print(f"Successfully wrote map with {len(degradation_map)} image entries.")
        except Exception as e:
            print(f"Error writing JSON file: {e}")
    else:
        print("No data processed, skipping JSON file generation.")

if __name__ == "__main__":
    main()

