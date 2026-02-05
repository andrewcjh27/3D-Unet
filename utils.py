import os
import torch
import nibabel as nib
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def save_predictions_as_images(loader, model, folder="saved_images/", device="cuda"):
    os.makedirs(folder, exist_ok=True)
    model.eval()

    for i, batch in enumerate(loader):
        images = batch["image"].to(device=device)
        masks = batch["mask"].to(device=device)

        with torch.no_grad():
            logits = model(images)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

        images_np = images.cpu().numpy()
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for j in range(images.shape[0]):
            img_to_save = np.transpose(images_np[j, 0, :, :, :], (1, 2, 0))
            pred_to_save = np.transpose(preds_np[j, 0, :, :, :], (1, 2, 0))
            mask_to_save = np.transpose(masks_np[j, 0, :, :, :], (1, 2, 0))

            affine = np.eye(4)
            img_nii = nib.Nifti1Image(img_to_save, affine)
            pred_nii = nib.Nifti1Image(pred_to_save, affine)
            mask_nii = nib.Nifti1Image(mask_to_save, affine)

            nib.save(img_nii, os.path.join(folder, f"image_{i*loader.batch_size+j}.nii.gz"))
            nib.save(pred_nii, os.path.join(folder, f"prediction_{i*loader.batch_size+j}.nii.gz"))
            nib.save(mask_nii, os.path.join(folder, f"mask_{i*loader.batch_size+j}.nii.gz"))

        break
    
    model.train()
    print(f"Saved prediction images to {folder}")

if __name__ == '__main__':
    from model import UNet3D
    from dataloader import BrainMRIDataset, DataLoader

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_FILENAME = "test_checkpoint.pth.tor"

    print("Testing save and load checkpoint...")
    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_checkpoint(checkpoint, filename=CHECKPOINT_FILENAME)

    new_model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)

    load_checkpoint(torch.load(CHECKPOINT_FILENAME), new_model, new_optimizer)

    print("Checkpoint test passed!")

    print("\nTesting save predictions as images...")

    os.makedirs('dummy_data/test/images', exist_ok=True)
    os.makedirs('dummy_data/test/masks', exist_ok=True)
    img = np.random.rand(128, 128, 64).astype(np.float32)
    mask = (np.random.rand(128, 128, 64) > 0.5).astype(np.float32)
    nib.save(nib.Nifti1Image(img, np.eye(4)), 'dummy_data/test/images/test_img.nii.gz')
    nib.save(nib.Nifti1Image(mask, np.eye(4)), 'dummy_data/test/masks/test_mask.nii.gz')

    test_dataset = BrainMRIDataset(
        image_dir='dummy_data/test/images',
        mask_dir='dummy_data/test/masks',
        patch_size=(128, 128, 64)
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    save_predictions_as_images(test_loader, new_model, device=DEVICE)

    print("Save predictions test finished. Check the 'saved_images' folder.")
