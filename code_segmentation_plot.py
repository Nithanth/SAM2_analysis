#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:21:58 2025

@author: mingjunsun
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils

def plot_single_coco_segmentation(image_path, ann_file, category_ids=None, save_path=None):
    """
    Overlays COCO-style segmentations for one image, when your JSON is either:
      - a single annotation dict
      - a list of annotation dicts, or
      - a dict with key 'annotations' → list of dicts.
    """
    # 1) load JSON
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # 2) normalize to a list of annotation dicts
    if isinstance(data, dict):
        if 'annotations' in data:
            anns = data['annotations']
        else:
            # assume it's a single annotation dict
            anns = [data]
    elif isinstance(data, list):
        anns = data
    else:
        raise ValueError(
            f"Unsupported JSON root type {type(data)}; "
            "expected dict-with-annotations, list, or single-annotation dict."
        )

    # 3) optional category filtering
    if category_ids is not None:
        anns = [ann for ann in anns if ann.get('category_id') in category_ids]

    # 4) load image & get its size
    img = np.array(Image.open(image_path))
    height, width = img.shape[:2]

    # 5) plot image
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    ax = plt.gca()

    # 6) overlay each segmentation
    for ann in anns:
        seg = ann.get('segmentation')

        # compressed RLE
        if isinstance(seg, dict):
            rle = seg

        # uncompressed RLE: list-of-dicts
        elif isinstance(seg, list) and seg and isinstance(seg[0], dict):
            rle = mask_utils.frPyObjects(seg, height, width)

        # polygon: list-of-lists-of-coordinates
        elif isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2)
                ax.plot(pts[:, 0], pts[:, 1], linewidth=2)
            continue

        else:
            raise ValueError(f"Unsupported segmentation format: {type(seg)}")

        # decode RLE and overlay mask
        m = mask_utils.decode(rle)
        ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.4)

    ax.axis('off')
    plt.tight_layout()

    # 7) save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    IMAGE_PATH = "/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic/93.jpg"
    ANN_FILE   = "/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic/93_annotations.json"
    SAVE_PATH  = "/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/93_seg.jpg"

    plot_single_coco_segmentation(IMAGE_PATH, ANN_FILE, save_path=SAVE_PATH)
