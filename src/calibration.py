"""Gender-conditional calibration using isotonic regression."""

import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from src.dataset import (
    load_dataframes, split_train_val, get_kfold_splits,
    FaceOcclusionDataset, get_val_transforms, DEFAULT_DATA_ROOT, _image_dir,
)
from src.model import FaceOcclusionModel
from src.utils import compute_score


def fit_gender_calibrators(preds, targets, genders):
    """Fit isotonic calibrators per gender on validation predictions.

    Args:
        preds: np.array of predicted occlusion scores
        targets: np.array of ground truth occlusion
        genders: np.array (0.0=F, 1.0=M)

    Returns:
        dict with keys "F" and "M", each an IsotonicRegression object.
    """
    mask_f = genders == 0.0
    mask_m = genders == 1.0

    iso_f = IsotonicRegression(out_of_bounds="clip")
    iso_f.fit(preds[mask_f], targets[mask_f])

    iso_m = IsotonicRegression(out_of_bounds="clip")
    iso_m.fit(preds[mask_m], targets[mask_m])

    return {"F": iso_f, "M": iso_m}


def apply_calibration(preds, pred_genders, calibrators):
    """Apply gender-conditional calibration.

    Args:
        preds: np.array of predicted occlusion scores
        pred_genders: np.array of bool (True=Male, False=Female)
        calibrators: dict from fit_gender_calibrators

    Returns:
        calibrated predictions (np.array)
    """
    calibrated = np.copy(preds)
    mask_m = pred_genders.astype(bool)
    mask_f = ~mask_m

    if mask_f.any():
        calibrated[mask_f] = calibrators["F"].transform(preds[mask_f])
    if mask_m.any():
        calibrated[mask_m] = calibrators["M"].transform(preds[mask_m])

    return calibrated


def save_calibrators(calibrators, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrators, f)
    print(f"Calibrators saved to {path}")


def load_calibrators(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit gender calibrators on validation set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a multi-task checkpoint (predict_gender=True)")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/submissions/checkpoints/calibrators.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None,
                        help="If set, use this fold's val split for calibration")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone_name", "convnext_tiny.fb_in22k_ft_in1k")
    predict_gender = ckpt.get("predict_gender", False)
    if not predict_gender:
        raise ValueError("Checkpoint must have predict_gender=True for calibration")

    model = FaceOcclusionModel(backbone_name=backbone_name, pretrained=False, predict_gender=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load val data using same split
    data_root = args.data_root or DEFAULT_DATA_ROOT
    image_dir = _image_dir(data_root)
    df_full, _ = load_dataframes(data_root=data_root)

    if args.fold is not None:
        splits = get_kfold_splits(df_full, n_splits=args.n_splits, seed=args.seed)
        _, val_idx = splits[args.fold]
        df_val = df_full.iloc[val_idx]
    else:
        _, df_val = split_train_val(df_full, seed=args.seed)

    val_ds = FaceOcclusionDataset(df_val, image_dir=image_dir, transform=get_val_transforms())
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Inference on val
    all_occ_preds = []
    all_targets = []
    all_genders = []

    with torch.no_grad():
        for imgs, targets, genders in val_loader:
            imgs = imgs.to(device)
            occ, _ = model(imgs)
            all_occ_preds.append(occ.cpu().numpy())
            all_targets.append(targets.numpy())
            all_genders.append(genders.numpy())

    preds = np.concatenate(all_occ_preds)
    targets = np.concatenate(all_targets)
    genders = np.concatenate(all_genders)

    # Score before calibration
    score_before, err_f_before, err_m_before = compute_score(preds, targets, genders)
    print(f"Before calibration: score={score_before:.6f} err_f={err_f_before:.6f} err_m={err_m_before:.6f}")

    # Fit calibrators using TRUE genders (available on val)
    calibrators = fit_gender_calibrators(preds, targets, genders)

    # Score after calibration (using true genders as proxy)
    pred_genders_true = (genders == 1.0)
    preds_cal = apply_calibration(preds, pred_genders_true, calibrators)
    score_after, err_f_after, err_m_after = compute_score(preds_cal, targets, genders)
    print(f"After calibration:  score={score_after:.6f} err_f={err_f_after:.6f} err_m={err_m_after:.6f}")

    # Save
    save_calibrators(calibrators, args.output)
