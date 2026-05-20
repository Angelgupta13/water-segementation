import torch


def get_metrics(preds, masks, threshold=0.5):
    """Returns dict of IOU, accuracy, precision, recall from logits vs binary masks."""
    preds = (torch.sigmoid(preds) > threshold).float()
    masks = masks.float()
    tp = (preds * masks).sum()
    fp = (preds * (1 - masks)).sum()
    fn = ((1 - preds) * masks).sum()
    tn = ((1 - preds) * (1 - masks)).sum()
    return {
        "iou":       (tp / (tp + fp + fn + 1e-8)).item(),
        "accuracy":  ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item(),
        "precision": (tp / (tp + fp + 1e-8)).item(),
        "recall":    (tp / (tp + fn + 1e-8)).item(),
    }


def iou_score(preds, masks, threshold=0.5):
    return get_metrics(preds, masks, threshold)["iou"]
