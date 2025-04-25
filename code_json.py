#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:50:33 2025

@author: mingjunsun
"""
import os
import json
import glob

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Base directory prefix to strip from all file paths in the JSON:
BASE_DIR        = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/'

# Directory containing the clean, renamed images ("1.jpg", "2.jpg", …)
PIC_DIR         = os.path.join(BASE_DIR, 'pic')
# Directory containing the per-degradation subfolders, each with files like "1_gaussian_blur_3.jpg"
DEG_DIR         = os.path.join(BASE_DIR, 'pic_degraded')
# Directory containing the per-image annotation JSONs ("1_annotations.json", etc.)
ANNO_DIR        = PIC_DIR
# Where to write the master JSON
OUTPUT_JSON     = os.path.join(BASE_DIR, 'degradation_path.json')

# List of degradation types
DEGRADATIONS = ['gaussian_blur', 'motion_blur', 'jpeg_compression', 'low_contrast']
# ── END CONFIG ─────────────────────────────────────────────────────────────────

# 1) Find all numeric image IDs from the PIC_DIR
image_ids = [
    os.path.splitext(fn)[0]
    for fn in os.listdir(PIC_DIR)
    if os.path.splitext(fn)[1].lower() in ('.jpg','.jpeg','.png')
       and os.path.splitext(fn)[0].isdigit()
]

# 2) Build up the data structure
data = {}
for img_id in image_ids:
    data[img_id] = { deg: [] for deg in DEGRADATIONS }
    data[img_id]['mask'] = None
    
    # a) Gather each degradation
    for deg in DEGRADATIONS:
        folder  = os.path.join(DEG_DIR, deg)
        pattern = os.path.join(folder, f"{img_id}_{deg}_*.jpg")
        for path in glob.glob(pattern):
            # extract the trailing number: "63_gaussian_blur_11.jpg" → "11"
            val = os.path.basename(path).rsplit('_', 1)[1].replace('.jpg','')
            # make it relative to BASE_DIR
            rel_path = os.path.relpath(path, BASE_DIR)
            # append as { "degradation_parameter": { "11": "pic_degraded/.../11.jpg" } }
            data[img_id][deg].append({
                'degradation_parameter': { val: rel_path }
            })
    
    # b) Load the mask segmentation from the annotation JSON
    anno_path = os.path.join(ANNO_DIR, f"{img_id}_annotations.json")
    if os.path.exists(anno_path):
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
        if isinstance(annotations, dict):
            annotations = [annotations]
        elif not isinstance(annotations, list):
            raise ValueError(f"Unexpected JSON format in {anno_path!r}")
        segs = [
            obj['segmentation']
            for obj in annotations
            if 'segmentation' in obj and obj['segmentation']
        ]
        first_poly = None
        for ann in annotations:
            seg = ann.get('segmentation', [])
            if seg:
                # seg is usually a list of polygons, so take the first one
                first_poly = seg[0] if isinstance(seg[0], list) else seg
                break
        
        # assign into your data structure
        data[img_id]['mask'] = first_poly

    else:
        data[img_id]['mask'] = []
        print(f"[Warning] annotation missing for image {img_id}")

# 3) Write everything out
with open(OUTPUT_JSON, 'w') as fo:
    json.dump(data, fo, indent=2)

print(f"Written master JSON to {OUTPUT_JSON}")
