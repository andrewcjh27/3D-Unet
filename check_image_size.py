# check_image_size.py
import nibabel as nib
import os

# --- Point this to your image file ---
IMAGE_PATH = "dummy_data/test/images/HF_0010_302.nii.gz" 

if os.path.exists(IMAGE_PATH):
    image_nii = nib.load(IMAGE_PATH)
    print(f"The dimensions of your image are: {image_nii.shape} (Height, Width, Depth)")
else:
    print(f"Error: Image not found at {IMAGE_PATH}")
