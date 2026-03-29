"""Model evaluation"""

import torch

class MetricTracker:
    """Accumulates TP/FP/TN counts"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = (torch.sigmoid(logits) > self.threshold).float()

        self.tp += ((preds == 1) & (target == 1)).sum().item()
        self.fp += ((preds == 1) & (target == 0)).sum().item()
        self.tn += ((preds == 0) & (target == 0)).sum().item()
        self.fn += ((preds == 0) & (target == 1)).sum().item()

    def compute(self) -> dict:
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + eps)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
        }
    
    def __repr__(self) -> str:
        metrics = self.compute()
        return {
            f"F1={metrics['f1']:.4f} |"
            f"IoU={metrics['iou']:.4f} |"
            f"Precision={metrics['precision']:.4f} |"
            f"Recall={metrics['recall']:.4f} |"
            f"Accuracy={metrics['accuracy']:.4f}"
        }