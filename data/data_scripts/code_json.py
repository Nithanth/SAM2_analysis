"""Generate `degradation_map.json` for SAM2 pipeline.

Scans the `data/images/` directory plus sibling degradation folders
(`data/pic_degraded/<degradation_type>/`) and produces a JSON file with the
structure expected by `sam2_eval_pipeline.py`:

{
  "<image_id>": {
    "ground_truth_rle": {"size": [H, W], "counts": "..."},
    "versions": {
      "original":           {"filepath": "images/<id>.jpg", "level": 0, "degradation_type": "original"},
      "gaussian_blur_5":    {"filepath": "pic_degraded/gaussian_blur/<id>_gaussian_blur_5.jpg", "level": 5,  "degradation_type": "gaussian_blur"},
      ...
    }
  }
}

Run directly:
    python data/data_scripts/code_json.py

Requirements:
    pip install pycocotools pillow
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image
from pycocotools import mask as mask_utils

# -----------------------------------------------------------------------------
# Configuration (relative to project root)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"  # originals (renamed 1.jpg, 2.jpg, ...)
DEG_DIR = DATA_DIR / "pic_degraded"  # sub-dirs per degradation type
ANNO_DIR = DATA_DIR / "image_annotations"  # *_annotations.json from COCO
OUTPUT_JSON = DATA_DIR / "degradation_map.json"

# Degradation types we expect as sub-directories inside DEG_DIR
DEGRADATIONS = [d.name for d in DEG_DIR.iterdir() if d.is_dir()]
print("Detected degradation types:", DEGRADATIONS)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_POLY_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(?:,?\s*)$")

def load_first_polygon(anno_path: Path) -> List[float]:
    """Return first polygon list `[x1,y1,x2,y2,...]` from annotation JSON."""
    with open(anno_path, "r") as f:
        data = json.load(f)

    # COCO files sometimes a dict or list
    anns = (
        data["annotations"]
        if isinstance(data, dict) and "annotations" in data
        else data if isinstance(data, list) else [data]
    )

    for ann in anns:
        seg = ann.get("segmentation")
        if seg:
            # segmentation can be list(list) for polygon or RLE dict
            if isinstance(seg, list):
                return seg[0] if isinstance(seg[0], list) else seg
            elif isinstance(seg, dict):
                # Already compressed RLE – return empty, we will use as-is later
                return []
    return []


def coco_poly_to_rle(segmentation: List, height: int, width: int) -> Dict:
    """Convert COCO polygon segmentation format to Run-Length Encoding (RLE).

    Args:
        segmentation: List of polygon coordinates [x1, y1, x2, y2, ...].
        height: Image height.
        width: Image width.

    Returns:
        RLE dictionary {'counts': [...], 'size': [h, w]}.
    """
    if not segmentation:
        return {}
    rle_list = mask_utils.frPyObjects([segmentation], height, width)
    rle = mask_utils.merge(rle_list) if isinstance(rle_list, list) else rle_list
    # `counts` may be bytes; convert to str for JSON serialisation
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {"size": [height, width], "counts": rle["counts"]}


def build_degradation_map() -> Dict[str, Dict]:
    """Construct the main degradation map dictionary.

    Iterates through sequentially numbered images in `pic/`, finds their
    original path, ground truth annotation (converted to RLE), and all
    corresponding degraded versions.

    Returns:
        A dictionary where keys are base image identifiers (e.g., '1') and
        values are dictionaries containing 'image_path', 'gt_mask_rle', and
        a flat 'versions' dictionary.
    """
    degradation_map = {}

    # Iterate through original, sequentially named images (e.g., 1.jpg, 2.jpg)
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img_id = img_path.stem  # assumes numeric or unique string

        # Load image size
        with Image.open(img_path) as im:
            w, h = im.size  # PIL gives (width, height)

        # --- Ground Truth Mask --- Check if annotation file exists
        anno_path = ANNO_DIR / f"{img_id}_annotations.json"
        if not anno_path.exists():
            print(f"[Warning] Annotation missing for {img_id}, skipping.")
            continue

        # Load annotation JSON
        with open(anno_path, "r") as f:
            data = json.load(f)

        # COCO files sometimes a dict or list
        anns = (
            data["annotations"]
            if isinstance(data, dict) and "annotations" in data
            else data if isinstance(data, list) else [data]
        )

        # Extract segmentation data (assuming it's polygon format)
        for ann in anns:
            seg = ann.get("segmentation")
            if seg:
                # segmentation can be list(list) for polygon or RLE dict
                if isinstance(seg, list):
                    polygon = seg[0] if isinstance(seg[0], list) else seg
                elif isinstance(seg, dict):
                    # Already compressed RLE – return empty, we will use as-is later
                    polygon = []
                break

        # Convert polygon segmentation to RLE format
        gt_rle = coco_poly_to_rle(polygon, h, w) if polygon else {}

        # --- Degraded Versions --- Find all corresponding degraded images
        versions: Dict[str, Any] = {
            "original": {
                "filepath": str(Path("images") / img_path.name),
                "level": 0,
                "degradation_type": "original",
            }
        }

        # Scan degradation folders
        for deg_type in DEGRADATIONS:
            folder = DEG_DIR / deg_type
            pattern = f"{img_id}_{deg_type}_*.jpg"
            for file in folder.glob(pattern):
                # extract level from filename using split on last underscore
                level_str = file.stem.rsplit("_", 1)[1]
                try:
                    level = float(level_str) if "." in level_str else int(level_str)
                except ValueError:
                    print(f"[Warning] Could not parse level from {file.name}")
                    continue
                key = f"{deg_type}_{level_str}"
                versions[key] = {
                    "filepath": str(Path("pic_degraded") / deg_type / file.name),
                    "level": level,
                    "degradation_type": deg_type,
                }

        # --- Assemble Entry --- Add the complete entry for this base image ID
        degradation_map[img_id] = {
            "ground_truth_rle": gt_rle,
            "versions": versions,
        }

    return degradation_map


def main() -> None:
    """Script entry point: build map and save to JSON."""
    # Build degradation map
    degradation_map = build_degradation_map()

    # Define the output path for the JSON map
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Save the map to the JSON file
    with open(OUTPUT_JSON, "w") as fo:
        json.dump(degradation_map, fo, indent=2)
    print(f"Written {OUTPUT_JSON.relative_to(ROOT_DIR)} with {len(degradation_map)} images.")


if __name__ == "__main__":
    main()
