#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:33:35 2025

@author: mingjunsun
"""

#%%
from pycocotools.coco import COCO
import requests
import os
import json
import random
import numpy as np

# Path to annotation file
annotation_file = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/annotations/instances_train2017.json'

# Initialize COCO API
coco = COCO(annotation_file)

# Find all image ids that have exactly one annotation with a non‐empty 'segmentation'
valid_img_ids = []
for img_id in coco.getImgIds():
    # 1) get all annotation IDs for this image
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    if not ann_ids:
        # no annotations at all → skip
        continue

    # 2) load all the ann dicts
    anns = coco.loadAnns(ann_ids)

    # 3) turn each ann into a single H×W mask
    masks = [ coco.annToMask(ann) for ann in anns ]

    # 4) union them into one mask (or create an empty one if none)
    if masks:
        unified = np.logical_or.reduce(masks).astype(np.uint8)
    else:
        img_info = coco.loadImgs([img_id])[0]
        h, w    = img_info['height'], img_info['width']
        unified = np.zeros((h, w), dtype=np.uint8)

    # 5) only keep images whose unified mask actually covers something
    if unified.sum() > 0:
        valid_img_ids.append(img_id)

# Sample up to 100 of these
n_samples = min(100, len(valid_img_ids))
sampled_img_ids = random.sample(valid_img_ids, n_samples)
images = coco.loadImgs(sampled_img_ids)

# Directories to save
save_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/images'
anno_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/image_annotations'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)

# Download images and save their single‐mask annotation
for img in images:
    # Download image
    img_data = requests.get(img['coco_url']).content
    img_path = os.path.join(save_dir, img['file_name'])
    with open(img_path, 'wb') as handler:
        handler.write(img_data)

    # Load the single annotation
    ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    ann = coco.loadAnns(ann_ids)[0]

    # Save that one annotation
    ann_path = os.path.join(anno_dir, img['file_name'].replace('.jpg', '.json'))
    with open(ann_path, 'w') as f:
        json.dump(ann, f, indent=2)

    print(f"Saved: {img['file_name']} with 1 segmentation mask")

#%%
import os
save_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic'
os.makedirs(save_dir, exist_ok=True)


#%%
import os
import random
import shutil
from PIL import Image

# Directories
input_dir       = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/images'
anno_input_dir  = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/image_annotations'
output_dir      = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic'

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all image files
image_files = [
    f for f in os.listdir(input_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Sample 100 images
sampled_images = random.sample(image_files, 100)

for idx, filename in enumerate(sampled_images, start=1):
    # --- Copy & rename image ---
    input_path  = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{idx}.jpg")
    
    img = Image.open(input_path).convert('RGB')
    img.save(output_path, 'JPEG')
    
    # --- Copy & rename annotation ---
    base_name         = os.path.splitext(filename)[0]
    anno_filename     = base_name + '.json'
    anno_input_path   = os.path.join(anno_input_dir, anno_filename)
    anno_output_path  = os.path.join(output_dir, f"{idx}_annotations.json")
    
    if os.path.exists(anno_input_path):
        shutil.copyfile(anno_input_path, anno_output_path)
        print(f"[{idx}] Saved image and {idx}_annotations.json")
    else:
        print(f"[{idx}] WARNING: Annotation for {filename} not found.")



#%%

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

input_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic'  # original images
output_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/pic_degraded'
os.makedirs(output_dir, exist_ok=True)

param_grid = {
    "gaussian_blur": {
        "kernel_size": [3, 5, 11, 21, 31]  # odd integers only
    },
    "motion_blur": {
        "kernel_size": [5, 15, 25, 35, 45]  # odd integers only
    },
    "jpeg_compression": {
        "quality": [100, 80, 60, 40, 20]  # 0–100 (higher = better)
    },
    "low_contrast": {
        "factor": [1.0, 0.8, 0.6, 0.4, 0.2]  # 1.0 = no change
    },
}


for d in param_grid:
    os.makedirs(os.path.join(output_dir, d), exist_ok=True)

def apply_gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_motion_blur(img, kernel_size):
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0
    k /= kernel_size
    return cv2.filter2D(img, -1, k)

def apply_jpeg_compression(img, quality):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, 1)

def apply_low_contrast(img, factor):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhanced = ImageEnhance.Contrast(pil).enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

dispatch = {
    "gaussian_blur":   apply_gaussian_blur,
    "motion_blur":     apply_motion_blur,
    "jpeg_compression":apply_jpeg_compression,
    "low_contrast":    apply_low_contrast,
}

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base, ext = os.path.splitext(filename)
    
    try:
        i = int(base)  
    except ValueError:
        print(f"Skipping non-numeric filename: {filename}")
        continue

    img = cv2.imread(os.path.join(input_dir, filename))

    for d_type, params in param_grid.items():
        func = dispatch[d_type]

        for param_name, values in params.items():
            for val in values:
                degraded = func(img, val)

                out_name = f"{i}_{d_type}_{val}.jpg"
                out_path = os.path.join(output_dir, d_type, out_name)
                cv2.imwrite(out_path, degraded)



#%%
import os
import cv2
import matplotlib.pyplot as plt
import random

# Paths
input_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/images'
output_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/degraded'
degradations = ['gaussian_blur', 'motion_blur', 'jpeg_compression', 'low_contrast']

# Randomly select three image filenames
all_filenames = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
selected_filenames = random.sample(all_filenames, 3)

# Titles for the columns
titles = ['Original'] + [d.replace('_', ' ').title() for d in degradations]

# Plot
plt.figure(figsize=(20, 12))

for row_idx, filename in enumerate(selected_filenames):
    # Load original image
    orig_path = os.path.join(input_dir, filename)
    orig_img = cv2.imread(orig_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Load degraded images
    degraded_imgs = []
    for d in degradations:
        d_path = os.path.join(output_dir, d, filename)
        d_img = cv2.imread(d_path)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        degraded_imgs.append(d_img)

    # Combine original and degraded
    images_to_show = [orig_img] + degraded_imgs

    # Plot this row
    for col_idx, (img, title) in enumerate(zip(images_to_show, titles)):
        idx = row_idx * 5 + col_idx + 1
        plt.subplot(3, 5, idx)
        plt.imshow(img)
        if row_idx == 0:
            plt.title(title, fontsize=30)  # Increased font size
        plt.axis('off')

plt.tight_layout()
plt.show()


