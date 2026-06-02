import numpy as np
import torch
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_mse_loss(pred, target):
    """Custom weighted MSE loss matching the challenge metric."""
    w = 1.0 / 30.0 + target
    return (w * (pred - target) ** 2).sum() / w.sum()


def compute_score(preds, targets, genders):
    """
    Compute the challenge score.
    genders: 0.0 = Female, 1.0 = Male (numeric).
    Returns: score (lower is better), err_f, err_m
    """
    def weighted_err(p, gt):
        w = 1.0 / 30.0 + gt
        return (w * (p - gt) ** 2).sum() / w.sum()

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(genders, torch.Tensor):
        genders = genders.detach().cpu().numpy()

    preds = np.array(preds, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    genders = np.array(genders, dtype=np.float64)

    mask_f = genders == 0.0
    mask_m = genders == 1.0

    err_f = weighted_err(preds[mask_f], targets[mask_f]) if mask_f.any() else 0.0
    err_m = weighted_err(preds[mask_m], targets[mask_m]) if mask_m.any() else 0.0

    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)
    return float(score), float(err_f), float(err_m)
