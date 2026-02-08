import os
import nibabel as nib
import numpy as np

# --- Configuration ---
# These should match the folders you are using in test.py
IMAGE_DIR = "dummy_data/test/images"
MASK_DIR = "dummy_data/test/masks"

# --- Script ---
print("Scanning for images and creating dummy masks...")
os.makedirs(MASK_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.nii', '.nii.gz'))]

if not image_files:
    print(f"No images found in {IMAGE_DIR}. Please place your image files there.")
else:
    for image_filename in image_files:
        # Load the original image to get its properties
        image_path = os.path.join(IMAGE_DIR, image_filename)
        original_image_nii = nib.load(image_path)
        image_shape = original_image_nii.shape
        image_affine = original_image_nii.affine

        # Create a numpy array of zeros with the same shape
        dummy_mask_data = np.zeros(image_shape, dtype=np.float32)

        # Create a new NIfTI image for the mask
        dummy_mask_nii = nib.Nifti1Image(dummy_mask_data, image_affine)

        # Save the dummy mask with the same filename
        mask_path = os.path.join(MASK_DIR, image_filename)
        nib.save(dummy_mask_nii, mask_path)
        print(f"Created dummy mask for '{image_filename}'")

print("\nDummy mask creation finished.")