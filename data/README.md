# Data Directory

This directory is intended to store input data for the analysis pipelines.

- **`degradation_map.json`**: (Required for `sam2_eval` pipeline) The JSON file mapping image IDs to ground truth RLE masks and image versions (potentially representing different degradation levels).
- **`images/`**: (Required for `sam2_eval` pipeline) Place the actual image files referenced in `degradation_map.json` here. The structure within this directory should match the relative paths specified in the `filepath` field under `versions` in the JSON map.

**Note:** This directory is ignored by git (see `.gitignore`). Do not commit large data files to the repository.
