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
    Overlays COCO-style segmentations for one image, when your JSON is just a list
    of 'annotation' dicts (not a full COCO dict).

    Parameters:
    -----------
    image_path : str
        Path to the image file.
    ann_file : str
        Path to a JSON file that is either:
          - a list of annotation dicts, or
          - a dict with key 'annotations'->list of annotation dicts.
    category_ids : list[int], optional
        If provided, only draw anns whose 'category_id' is in this list.
    save_path : str, optional
        If provided, save the resulting plot here; otherwise plt.show().
    """
    # load JSON
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # extract annotations
    if isinstance(data, dict) and 'annotations' in data:
        anns = data['annotations']
    elif isinstance(data, list):
        anns = data
    else:
        raise ValueError("ann_file must be either a list or a dict with 'annotations' key.")

    # optional category filtering
    if category_ids is not None:
        anns = [ann for ann in anns if ann.get('category_id') in category_ids]

    # load image & get its size
    img = np.array(Image.open(image_path))
    height, width = img.shape[:2]

    # plot
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    ax = plt.gca()

    for ann in anns:
        seg = ann['segmentation']
        if isinstance(seg, list):
            # polygons
            for poly in seg:
                pts = np.array(poly).reshape(-1,2)
                ax.plot(pts[:,0], pts[:,1], linewidth=2)
        else:
            # RLE
            rle = seg if isinstance(seg, dict) else mask_utils.frPyObjects(seg, height, width)
            m = mask_utils.decode(rle)
            ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.4)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


IMAGE_PATH = "/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic/100.jpg"
ANN_FILE   = "/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic/100_annotations.json"
SAVE_PATH  = "/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/100_seg.jpg"


# Only show COCO cats 1 and 3:
# plot_single_coco_segmentation(IMAGE_PATH, ANN_FILE, category_ids=[1,3], save_path=SAVE_PATH)

plot_single_coco_segmentation(IMAGE_PATH, ANN_FILE, save_path=SAVE_PATH)
