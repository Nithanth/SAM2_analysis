"""
LoRA fine-tuning for SAM2 THIS IS NOT COMPLETE LOL HIERA IS HARD

This script adds Low-Rank Adaptation (LoRA) adapters to the SAM2 image
encoder (Hiera backbone).
"""

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import requests  # For downloading sample image
import argparse
import os
from tqdm import tqdm

try:
    from sam2 import SAM2ImagePredictor
except ImportError as e:
    raise ImportError(
        "sam2 library not found. Clone facebookresearch/sam2 and install with "
        "`pip install -e .` before running this script."
    ) from e

from peft import LoraConfig, get_peft_model


MODEL_ID = "facebook/sam2-hiera-tiny" # Or other SAM2 variant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Base SAM2 Model 
def load_base_model(model_id: str = MODEL_ID, device: str = DEVICE):
    """Returns (image_encoder_module, original_predictor)."""
    print(f"Loading SAM2 model {model_id} …")
    predictor = SAM2ImagePredictor.from_pretrained(model_id=model_id, device=device)

    # We only adapt the image encoder (Hiera backbone)
    if not hasattr(predictor, "image_encoder"):
        raise AttributeError("SAM2ImagePredictor has no attribute `image_encoder`; "
                             "unable to attach LoRA adapters.")
    model = predictor.image_encoder
    model.to(device)
    model.eval()
    print("Image encoder ready for LoRA.")
    return model, predictor

# Define LoRA Configuration 
def create_lora_config(model: torch.nn.Module) -> LoraConfig:
    """Selects attention projection layers inside *model* as LoRA targets."""
    candidate_suffixes = ("q_proj", "k_proj", "v_proj", "qkv")
    targets = {name.split(".")[-1] for name, _ in model.named_modules()
               if name.endswith(candidate_suffixes)}
    if not targets:
        targets = {"q_proj", "k_proj", "v_proj"}  # reasonable default
    print(f"LoRA will be inserted into: {sorted(targets)}")
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=list(targets),
        lora_dropout=0.05,
        bias="none",
    )

# Apply LoRA to the Model 
def apply_lora(model, lora_config):
    print("Applying LoRA adapters to the model...")
    try:
        lora_model = get_peft_model(model, lora_config)
        print("LoRA applied successfully.")
        lora_model.print_trainable_parameters()
        return lora_model
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        print("This might be due to incorrect target_modules or incompatibilities.")
        return None

# Helper: Placeholder Dataset 
class FineTuneDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # todo: Load image paths and GT mask info from data_path
        # Example: self.items = parse_metadata(data_path)
        self.items = [("sample_url", "dummy_gt")] * 10 # Placeholder
        self.transform = transform # Image transforms (resize, normalize)
        self.use_sample_url = True # Flag to use sample download
        print(f"Warning: Using placeholder {self.__class__.__name__}.")
        print("Replace with your actual dataset loading logic.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_source, gt_source = self.items[idx]

        # TODO: Load actual image and GT mask
        if self.use_sample_url:
            url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
            try:
                image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            except Exception as e:
                print(f"Error loading sample image: {e}")
                # Return dummy data on error
                image = Image.new('RGB', (256, 256), color = 'red')
        else:
            # image = Image.open(img_source).convert("RGB")
            pass # Implement real image loading

        # Create a dummy ground truth mask (e.g., a square in the middle)
        # TODO: Load/decode your actual GT mask (e.g., from RLE or file)
        gt_mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
        cx, cy = image.width // 2, image.height // 2
        size = min(image.width, image.height) // 4
        gt_mask_np[cy-size:cy+size, cx-size:cx+size] = 1
        gt_mask = torch.tensor(gt_mask_np, dtype=torch.float32).unsqueeze(0) # Add channel dim

        # TODO: Apply necessary transforms (resizing, normalization)
        #       matching how SAM2 expects input.
        # Example using hypothetical transform:
        # if self.transform:
        #     image = self.transform(image)

        # Return image (as tensor) and mask (as tensor)
        return image, gt_mask

# ugh fig this out
def loss(pred, target, smooth=1.):
    pass

def train_one_epoch(lora_model, original_predictor, dataloader, optimizer, loss_fn, device):
    lora_model.train() # Set the image encoder (wrapped by PEFT) to train mode
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for batch_idx, (images, gt_masks) in enumerate(progress_bar):
        # TODO: Convert images to tensor format expected by SAM2/Hiera + to(device)
        # input_images = preprocess_batch(images).to(device)
        # gt_masks = gt_masks.to(device)

        try:
           
            # Placeholder: Generate dummy predictions 
            print(f"Warning: Using DUMMY predictions in training step {batch_idx}. Replace forward pass.")
            pred_masks = torch.rand_like(gt_masks)
            
            loss = loss_fn(pred_masks, gt_masks)

            # --- Backward Pass & Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        except Exception as e:
            print(f"Error during training step {batch_idx}: {e}")
            print("Skipping batch. Forward pass likely needs correction.")
            # Optionally: raise e to debug
            continue

    avg_loss = total_loss / len(dataloader)
    print(f"Training Epoch Finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(lora_model, original_predictor, dataloader, loss_fn, device):
    lora_model.eval() # Set encoder to evaluation mode
    total_loss = 0
    # TODO: Add metric calculations (mIoU, BF1) using functions from metrics.py
    # total_miou = 0

    progress_bar = tqdm(dataloader, desc="Validation Epoch", leave=False)
    with torch.no_grad():
        for batch_idx, (images, gt_masks) in enumerate(progress_bar):
            # TODO: Convert images to tensor format expected by SAM2/Hiera + to(device)
            # input_images = preprocess_batch(images).to(device)
            # gt_masks = gt_masks.to(device)

            try:
                # Placeholder: Generate dummy predictions 
                print(f"Warning: Using DUMMY predictions in validation step {batch_idx}. Replace forward pass.")
                pred_masks = torch.rand_like(gt_masks)
                
                loss = loss_fn(pred_masks, gt_masks)
                total_loss += loss.item()

                # todo: Calculate and accumulate metrics
                # batch_miou = calculate_miou(pred_masks.cpu().numpy(), gt_masks.cpu().numpy())
                # total_miou += batch_miou

            except Exception as e:
                print(f"Error during validation step {batch_idx}: {e}")
                continue

    avg_loss = total_loss / len(dataloader)
    # avg_miou = total_miou / len(dataloader)
    print(f"Validation Epoch Finished. Average Loss: {avg_loss:.4f}")
    # print(f"Validation mIoU: {avg_miou:.4f}") # TODO
    return avg_loss # , avg_miou

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 Image Encoder with LoRA.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data metadata or directory.")
    # parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data.")
    parser.add_argument("--output_dir", type=str, default="./sam2-lora-adapters", help="Directory to save LoRA adapters.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Base SAM2 model ID.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load Base Model (Image Encoder) and Original Predictor
    base_model, predictor_instance = load_base_model(model_id=args.model_id, device=DEVICE)
    if base_model is None:
        print("Exiting due to model loading failure.")
        exit()

    # Create LoRA Config
    # Update config based on argparse if needed
    lora_config = create_lora_config(base_model)
    lora_config.r = args.lora_r
    lora_config.lora_alpha = args.lora_alpha
    print(f"Using LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, targets={lora_config.target_modules}")

    # Apply LoRA to the Image Encoder
    lora_model = apply_lora(base_model, lora_config)
    if lora_model is None:
        print("Exiting due to LoRA application failure.")
        exit()

    # Setup DataLoaders 
    # still todo: Define appropriate transforms
    train_transform = None 
    # val_transform = None
    train_dataset = FineTuneDataset(args.data_path, transform=train_transform)
    # val_dataset = FineTuneDataset(args.val_data_path, transform=val_transform) 
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Loaded placeholder training data with {len(train_dataset)} samples.")
    # print(f"Loaded placeholder validation data with {len(val_dataset)} samples.")

    # Setup Optimizer and Loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=args.lr)
    loss_function = dice_loss

    print("\n--- Starting LoRA Fine-tuning ---")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # --- Training --- 
        train_loss = train_one_epoch(
            lora_model,
            predictor_instance, # Pass original predictor for its components
            train_dataloader,
            optimizer,
            loss_function,
            DEVICE
        )
        
        # --- Validation --- 
        val_loss = evaluate(
            lora_model, 
            predictor_instance, 
            val_dataloader, 
            loss_function, 
            DEVICE
        )

        # --- Save Adapters --- 
        epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        print(f"Saving LoRA adapters for epoch {epoch+1} to {epoch_output_dir}")
        lora_model.save_pretrained(epoch_output_dir)

    print("\n--- LoRA Fine-tuning Finished ---")
    print(f"Final adapters saved in {args.output_dir}")


