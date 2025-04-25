# SAM2_analysis
10623
code_degradation.py: download images and annotations, degrade each image by four types and each type has 5 degradation parameters.
code_json.py: generate a json document 
code_segmentation_plot.py: plot the segmentation

## Setup

This project analyzes the robustness of the SAM2 model. Follow these steps to set up the environment:

1. **Clone the Repository (with Submodule):**

SAM2 is included as a Git submodule. Clone the repository using the `--recurse-submodules` flag to ensure the SAM2 code is downloaded:

```bash
git clone --recurse-submodules https://github.com/<your_github_username>/SAM2_analysis.git # Replace with your actual repo URL
cd SAM2_analysis
```

Alternatively, if you have already cloned without the flag, navigate into the repository and run:

```bash
git submodule update --init --recursive
```

2. **Create a Virtual Environment (Recommended):**

It's highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
```

3. **Install Dependencies:**

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: Ensure your environment has the correct PyTorch version compatible with your hardware (CPU/GPU + CUDA).*

4. **Install SAM2 Submodule Package:**

Install the downloaded SAM2 code as an editable package so Python can find it:

```bash
pip install -e ./external/sam2
```

5. **(Alternative) Hugging Face `transformers`:**

If SAM2 becomes officially integrated into the Hugging Face `transformers` library in the future, you might be able to use it directly through `transformers` without relying on the submodule. This would typically involve installing `transformers` and using its specific classes for SAM2.

## Usage

(Instructions on how to run the analysis, inference scripts, etc., will be added here later.)
