"""
Metrics for binary segmentation.
All four metrics required by assignment: IOU, Accuracy, Precision, Recall.
"""
import torch


def get_metrics(preds, masks, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    masks = masks.float()

    tp = (preds * masks).sum()
    fp = (preds * (1 - masks)).sum()
    fn = ((1 - preds) * masks).sum()
    tn = ((1 - preds) * (1 - masks)).sum()

    iou       = tp / (tp + fp + fn + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    return {
        "iou":       iou.item(),
        "accuracy":  accuracy.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
    }


def iou_score(preds, masks, threshold=0.5):
    return get_metrics(preds, masks, threshold)["iou"]
