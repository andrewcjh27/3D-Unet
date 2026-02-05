import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib
import numpy as np

# Import the components we've built
from model import UNet3D
from dataloader import BrainMRIDataset
from loss import BCEDiceLoss
from metric import calculate_dice_score
from utils import save_checkpoint, load_checkpoint

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate(loader, model, loss_fn, device):
    model.eval()
    running_loss = 0.0
    total_dice_score = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            logits = model(images)
            loss =loss_fn(logits, masks)

            running_loss += loss.item()
            total_dice_score += calculate_dice_score(logits, masks)

    avg_loss = running_loss / len(loader)
    avg_dice = total_dice_score / len(loader)
    return avg_loss, avg_dice
    
def main():
    DEVICE = "cude" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    NUM_WORKERS = 2
    PATCH_SIZE = (128, 128, 64)

    TRAIN_IMG_DIR = "dummy_data/train/images"
    TRAIN_MASK_DIR = "dummy_data/train/masks"
    VAL_IMG_DIR = "dummy_data/val/images"
    VAL_MASK_DIR = "dummy_data/val/masks"

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")

    # --- CHANGE 2: The dummy data creation block has been removed. ---
    # You should delete the corresponding block from your file.

    # --- Initialize Model, Loss, Optimizer, and DataLoaders ---
    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = BrainMRIDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, patch_size=PATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    val_dataset = BrainMRIDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, patch_size=PATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # --- Training Loop ---
    best_val_dice = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss, val_dice = validate(val_loader, model, loss_fn, DEVICE)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Dice Score: {val_dice:.4f}")

        # Save the model if it has the best validation Dice score so far
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=BEST_MODEL_PATH)
            print(f"New best model saved to {BEST_MODEL_PATH}")

    print("\nTraining finished!")

if __name__ == "__main__":
    main()
