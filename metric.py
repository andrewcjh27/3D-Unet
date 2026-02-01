import torch

def calculate_dice_score(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    total_sum = preds.sum() + targets.sum()

    dice_score = (2. * intersection +smooth) / (total_sum + smooth)

    return dice_score.item()

if __name__ ==  '__main__':

    batch_size = 4
    num_classes = 1
    patch_depth, patch_height, patch_width = 64, 128, 128

    dummy_logits = torch.randn(batch_size, num_classes, patch_depth, patch_height, patch_width)

    dummy_targets = (torch.rand(batch_size, num_classes, patch_depth, patch_height, patch_width) > 0.5).float()

    print(f"Input logits shape: {dummy_logits.shape}")
    print(f"Input targets shape: {dummy_targets.shape}")

    print("\nTesting caculate_dice_score...")
    dice_score_val = calculate_dice_score(dummy_logits, dummy_targets)
    print(f"Dice Score: {dice_score_val}")
    assert 0 <= dice_score_val <= 1, "Dice Score should be between 0 and 1."
    print("Metric test pased!")