import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        #apply sigmoid
        probs = torch.sigmoid(logits)

        #flatten the tensors to treat each voxel as a sample
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        #dice loss is 1 - dice coefficient
        return 1 - dice_coeff
    
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)

        combined_loss =self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss

if __name__ == '__main__':

    batch_size =4
    num_classes = 1
    patch_depth, patch_height, patch_width = 64, 128, 128

    dummy_logits = torch.randn(batch_size, num_classes, patch_depth, patch_height, patch_width)

    dummy_targets = (torch.rand(batch_size, num_classes, patch_depth, patch_height, patch_width) > 0.5).float()

    print(f"Input logits shape: {dummy_logits.shape}")
    print(f"Input targets shape: {dummy_targets.shape}")

    print("\nTesting DiceLoss...")

    dice_loss_fn = DiceLoss()
    dice_loss_val = dice_loss_fn(dummy_logits, dummy_targets)
    print(f"Dice Loss: {dice_loss_val.item()}")
    assert 0 <= dice_loss_val.item() <= 1
    print("DICELoss test passed!")

    print("\nTesting BCEDiceLoss...")
    bce_dice_loss_fn =BCEDiceLoss()
    bce_dice_loss_val = bce_dice_loss_fn(dummy_logits, dummy_targets)
    print(f"BCE Dice Loss: {bce_dice_loss_val.item()}")
    print("BCEDiceLoss test passed!")

