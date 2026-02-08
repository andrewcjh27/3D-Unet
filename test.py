import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib
import numpy as np

# Import the components we've built
from model import UNet3D
from dataloader import BrainMRIDataset
from metric import calculate_dice_score
from utils import load_checkpoint, save_predictions_as_images

def check_accuracy(loader, model, device="cuda"):
    print("Checking accuracy on test set...")
    model.eval()
    total_dice_score = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            total_dice_score += calculate_dice_score(logits, masks)

    avg_dice = total_dice_score / len(loader)
    print(f"Test Set Dice Score: {avg_dice:.4f}")
    return avg_dice

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 2
    NUM_WORKERS = 2
    PATCH_SIZE = (512, 512, 55)

    TEST_IMG_DIR = "dummy_data/test/images"
    TEST_MASK_DIR = "dummy_data/test/masks"

    CHECKPOINT_PATH = "checkpoints/best_model.pth.tar"
    SAVED_PREDICTIONS_DIR = "saved_predictons/"

    '''print("Creating dummy test data for demonstration...")
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_MASK_DIR, exist_ok=True)
    for i in range(4):
        img = np.random.rand(256, 256, 150).astype(np.float32)
        mask = (np.random.rand(256, 256, 150) > 0.5).astype(np.float32)
        nib.save(nib.Nifti1Image(img, np.eye(4)), os.path.join(TEST_IMG_DIR, f"test_{i}.nii.gz"))
        nib.save(nib.Nifti1Image(mask, np.eye(4)), os.path.join(TEST_MASK_DIR, f"test_{i}.nii.gz"))
    print("Dummy test data created.")'''

    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
                
    optimizer = torch.optim.Adam(model.parameters())

    if os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(torch.load(CHECKPOINT_PATH), model, optimizer)
    else: 
        print("Checkpoint file not found. Using a randomly initialized model for testing.")

    test_dataset = BrainMRIDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, patch_size=PATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    check_accuracy(test_loader, model, device=DEVICE)

    print("\nSaving some predictions for visual inspection...")
    save_predictions_as_images(
        test_loader, model, folder=SAVED_PREDICTIONS_DIR, device=DEVICE
    )
    print(f"Predictions saved to '{SAVED_PREDICTIONS_DIR}' folder.")

if __name__ == "__main__":
    main()




