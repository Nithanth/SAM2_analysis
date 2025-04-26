# Data Directory

This directory is intended to store input data for the analysis pipelines.

- **`degradation_map.json`**: (Required for `sam2_eval` pipeline) The JSON file mapping image IDs to ground truth RLE masks and image versions (potentially representing different degradation levels). It uses a nested structure for versions:

  ```json
  {
    "<image_id>": {
      "ground_truth_rle": {
        "size": [H, W],    # Mask dimensions [height, width]
        "counts": "..."     # COCO Run-Length Encoding string for the mask pixels
      },
      "versions": {
        "original":           {"filepath": "images/<id>.jpg", "level": 0, "degradation_type": "original"},
        "gaussian_blur": {
          "5": {"filepath": "pic_degraded/gaussian_blur/<id>_gaussian_blur_5.jpg", "level": 5,  "degradation_type": "gaussian_blur"},
          ...
        },
        "jpeg_compression": {
          "80": {"filepath": "pic_degraded/jpeg_compression/<id>_jpeg_compression_80.jpg", "level": 80,  "degradation_type": "jpeg_compression"}
        }
      }
    },
    ...
  }
  ```

- **`images/`**: (Required for `sam2_eval` pipeline) Place the actual image files referenced in `degradation_map.json` here. The structure within this directory should match the relative paths specified in the `filepath` field under the nested `versions` in the JSON map.

**Note:** This directory is ignored by git (see `.gitignore`). Do not commit large data files to the repository.
