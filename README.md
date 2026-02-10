# 3D U-Net

PyTorch implementation of a 3D U-Net for volumetric brain MRI segmentation. Trained on NIfTI (.nii.gz) images with patch-based loading and BCE-Dice loss.

## Architecture

- 3D convolutional encoder-decoder with skip connections
- Patch-based data loading (128x128x64) for memory efficiency
- Combined BCE + Dice loss for handling class imbalance

## Files

| File | Description |
|------|-------------|
| `model.py` | 3D U-Net architecture |
| `dataloader.py` | NIfTI dataset with patch extraction |
| `train.py` | Training loop with checkpointing |
| `test.py` | Inference and evaluation |
| `loss.py` | BCE-Dice combined loss |
| `metric.py` | Dice score computation |
| `utils.py` | Checkpoint save/load utilities |

## Requirements

- PyTorch
- nibabel
- numpy
- tqdm

## Usage

```bash
python train.py
python test.py
```
