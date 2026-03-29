"""Loss functions"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth: float = 1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice
    
class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss"""

    def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            pos_weight: float = 3.0,):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss