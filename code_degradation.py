#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:33:35 2025

@author: mingjunsun
"""

#%%

'''
from pycocotools.coco import COCO
import requests
import os

# Path to the annotation file
annotation_file = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/annotations/instances_train2017.json'

# Initialize COCO api for instance annotations
coco = COCO(annotation_file)

# Get all image ids
img_ids = coco.getImgIds()
images = coco.loadImgs(img_ids)


# Directory to save images
save_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/images'
os.makedirs(save_dir, exist_ok=True)

# Download images

for img in images:
    img_data = requests.get(img['coco_url']).content
    with open(os.path.join(save_dir, img['file_name']), 'wb') as handler:
        handler.write(img_data)
'''


#%%
import os
save_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/images'
os.makedirs(save_dir, exist_ok=True)


#%%

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

# Input and output directories
input_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/images'  # original images
output_dir = '/Users/mingjunsun/Library/CloudStorage/Dropbox/25 Spring/Generative AI/project/dataset/degraded'
os.makedirs(output_dir, exist_ok=True)

# Subfolders for different types of degradation
degradations = ['gaussian_blur', 'motion_blur', 'jpeg_compression', 'low_contrast']
for d in degradations:
    os.makedirs(os.path.join(output_dir, d), exist_ok=True)

def apply_gaussian_blur(image, kernel_size=15):  # Increased kernel_size
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_motion_blur(image, kernel_size=25):  # Increased kernel_size
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_jpeg_compression(image, quality=5):  # Decreased quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_low_contrast(image, factor=0.3):  # Decreased factor
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    low_contrast_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(low_contrast_img), cv2.COLOR_RGB2BGR)

# Loop through all images and apply degradations
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)

        # Apply and save each degradation
        degraded = {
            'gaussian_blur': apply_gaussian_blur(image),
            'motion_blur': apply_motion_blur(image),
            'jpeg_compression': apply_jpeg_compression(image),
            'low_contrast': apply_low_contrast(image),
        }

        for d_type, d_img in degraded.items():
            out_path = os.path.join(output_dir, d_type, filename)
            cv2.imwrite(out_path, d_img)

