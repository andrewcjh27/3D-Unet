import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(128, 128, 64), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.transform = transform

        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.mask_files = sorted([os.path.join(mask_dir, f)  for f in os.listdir(mask_dir) if f.endswith(('.nii', '.nii.gz'))])
        assert len(self.image_files) == len(self.mask_files)
        assert len(self.image_files) > 0
                   
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        #load full 3D image and mask
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image = image_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        #randomly extract 3D patch
        img_h, img_w, img_d =image.shape
        patch_h, patch_w, patch_d = self.patch_size

        if patch_h > img_h or patch_w > img_w or patch_d > img_d:
            raise ValueError("Patch size is larger than the image dimensions.")
        
        h_start = np.random.randint(0, img_h - patch_h + 1)
        w_start = np.random.randint(0, img_w - patch_w + 1)
        d_start = np.random.randint(0, img_d - patch_d + 1)

        image_patch = image[h_start : h_start + patch_h, w_start : w_start + patch_w, d_start : d_start + patch_d]
        mask_patch = mask[h_start : h_start + patch_h, w_start : w_start + patch_w, d_start : d_start + patch_d]

        #processing and formatting
        image_patch = np.expand_dims(image_patch, axis=0)
        mask_patch = np.expand_dims(mask_patch, axis=0)

        # PyTorch expects channels first, so we transpose to (C, D, H, W)
        image_patch = image_patch.transpose(0, 3, 1, 2) # Shape: (1, D, H, W)
        mask_patch = mask_patch.transpose(0, 3, 1, 2)   # Shape: (1, D, H, W)

        # Convert numpy arrays to PyTorch tensors
        image_patch = torch.from_numpy(image_patch.copy())
        mask_patch = torch.from_numpy(mask_patch.copy())

        sample = {'image': image_patch, 'mask': mask_patch}

        if self.transform:
            sample = self.transform(sample)

        return sample
if __name__ == '__main__':
    print("Creating dummy data for testing...")
    os.makedirs('dummy_data/images', exist_ok=True)
    os.makedirs('dummy_data/masks', exist_ok=True)

    PATCH_SIZE = (128, 128, 64)
    BATCH_SIZE = 4

    for i in range(BATCH_SIZE):
        full_volume_shape = (256, 256, 150)

        dummy_image_data = np.random.rand(*full_volume_shape).astype(np.float32)
        dummy_mask_data = (np.random.rand(*full_volume_shape) >  0.5).astype(np.float32)

        affine = np.eye(4)
        dummy_image_nii = nib.Nifti1Image(dummy_image_data, affine)
        dummy_mask_nii = nib.Nifti1Image(dummy_mask_data, affine)

        nib.save(dummy_image_nii, f'dummy_data/images/image_{i}.nii.gz')
        nib.save(dummy_mask_nii, f'dummy_data/masks/mask_{i}.nii.gz')

    print("Dummy data created.")

    brain_dataset = BrainMRIDataset(
        image_dir='dummy_data/images',
        mask_dir='dummy_data/masks',
        patch_size=PATCH_SIZE
    )

    data_loader = DataLoader(
        dataset=brain_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    print("\nTesting the DataLoader...")
    batch = next(iter(data_loader))

    images = batch['image']
    masks = batch['mask']

    print(f"Batch size: {len(images)}")
    print(f"Image patch shape: {images.shape}")
    print(f"Mask patch shape: {masks.shape}")

    expected_shape = (BATCH_SIZE, 1, PATCH_SIZE[2],PATCH_SIZE[0], PATCH_SIZE[1])
    assert images.shape == expected_shape, f"Image shape is not correct. Expected {expected_shape}, got {images.shape}"
    assert masks.shape == expected_shape, f"Mask shape is not correct. Expected {expected_shape}, got {masks.shape}"

    print("\nDataloader test passed!")

