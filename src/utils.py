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


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _weighted_err(pred, target):
    """Weighted error for a single group (torch tensors)."""
    w = 1.0 / 30.0 + target
    return (w * (pred - target) ** 2).sum() / w.sum()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def weighted_mse_loss(pred, target, genders=None, male_factor=1.0):
    """Custom weighted MSE loss matching the challenge metric.

    If genders and male_factor != 1.0, male samples get extra weight.
    Backward-compatible: call without genders gives the original result.
    """
    w = 1.0 / 30.0 + target
    if genders is not None and male_factor != 1.0:
        w = w * torch.where(
            genders == 1.0,
            torch.as_tensor(male_factor, device=w.device, dtype=w.dtype),
            torch.ones_like(w),
        )
    return (w * (pred - target) ** 2).sum() / w.sum()


def metric_surrogate_loss(pred, target, genders):
    """Differentiable surrogate of the challenge score.

    Returns 0.5*(err_f + err_m) + |err_f - err_m|.
    Falls back to global weighted_mse if a gender has < 2 samples.
    """
    mask_f = (genders == 0.0)
    mask_m = (genders == 1.0)

    if mask_f.sum() < 2 or mask_m.sum() < 2:
        return weighted_mse_loss(pred, target)

    err_f = _weighted_err(pred[mask_f], target[mask_f])
    err_m = _weighted_err(pred[mask_m], target[mask_m])

    return 0.5 * (err_f + err_m) + torch.abs(err_f - err_m)


def groupdro_loss(pred, target, genders):
    """Group DRO loss: worst-group weighted error (max of err_f, err_m).

    Falls back to global weighted_mse if a gender has < 2 samples.
    """
    mask_f = (genders == 0.0)
    mask_m = (genders == 1.0)

    if mask_f.sum() < 2 or mask_m.sum() < 2:
        return weighted_mse_loss(pred, target)

    err_f = _weighted_err(pred[mask_f], target[mask_f])
    err_m = _weighted_err(pred[mask_m], target[mask_m])

    return torch.max(err_f, err_m)


def fairness_loss(pred, target, genders, fairness_lambda=1.0):
    """(Deprecated: prefer male_factor or groupdro_loss — see README.)

    Weighted MSE + fairness regularization penalizing the gender error gap.
    loss = weighted_mse + lambda * |weighted_mse_F - weighted_mse_M|
    """
    base_loss = weighted_mse_loss(pred, target)

    mask_f = (genders == 0.0)
    mask_m = (genders == 1.0)

    if mask_f.sum() < 2 or mask_m.sum() < 2:
        return base_loss

    err_f = _weighted_err(pred[mask_f], target[mask_f])
    err_m = _weighted_err(pred[mask_m], target[mask_m])

    return base_loss + fairness_lambda * torch.abs(err_f - err_m)


# ---------------------------------------------------------------------------
# Evaluation metric (DO NOT MODIFY BEHAVIOR)
# ---------------------------------------------------------------------------

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
