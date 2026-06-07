"""Diagnostics: error decomposition by gender × occlusion bin."""

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.dataset import (
    load_dataframes, split_train_val, get_kfold_splits,
    FaceOcclusionDataset, get_val_transforms, DEFAULT_DATA_ROOT, _image_dir,
)
from src.model import FaceOcclusionModel
from src.utils import compute_score


def _distribution_by_gender(df_full, output_dir="docs"):
    """Plot and save occlusion distribution by gender."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (g, label) in zip(axes, [(0, "Female"), (1, "Male")]):
        subset = df_full[df_full["gender"] == g]["FaceOcclusion"]
        ax.hist(subset, bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(f"{label} (n={len(subset)})")
        ax.set_xlabel("FaceOcclusion")
        ax.set_ylabel("Count")
        stats = subset.describe()
        ax.axvline(stats["mean"], color="red", linestyle="--", label=f"mean={stats['mean']:.4f}")
        ax.axvline(stats["50%"], color="green", linestyle="--", label=f"median={stats['50%']:.4f}")
        ax.legend()

    plt.suptitle("Occlusion distribution by gender", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "occlusion_distribution_by_gender.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Print describe
    print("\n=== Distribution by gender ===")
    for g, label in [(0, "Female"), (1, "Male")]:
        print(f"\n{label}:")
        print(df_full[df_full["gender"] == g]["FaceOcclusion"].describe())


def _error_decomposition(preds, targets, genders, n_bins=10):
    """Decompose error by gender × occlusion bin."""
    df = pd.DataFrame({
        "pred": preds, "target": targets, "gender": genders
    })
    df["occ_bin"] = pd.qcut(df["target"], q=n_bins, labels=False, duplicates="drop")
    df["weight"] = 1.0 / 30.0 + df["target"]
    df["weighted_sq_err"] = df["weight"] * (df["pred"] - df["target"]) ** 2

    rows = []
    for gender_val, gender_label in [(0.0, "F"), (1.0, "M")]:
        for occ_bin in sorted(df["occ_bin"].unique()):
            mask = (df["gender"] == gender_val) & (df["occ_bin"] == occ_bin)
            subset = df[mask]
            if len(subset) == 0:
                continue
            w_err = subset["weighted_sq_err"].sum() / subset["weight"].sum()
            occ_range = f"[{subset['target'].min():.3f}, {subset['target'].max():.3f}]"
            rows.append({
                "gender": gender_label,
                "bin": occ_bin,
                "occ_range": occ_range,
                "n_samples": len(subset),
                "weighted_err": w_err,
                "mean_pred": subset["pred"].mean(),
                "mean_target": subset["target"].mean(),
            })

    df_dec = pd.DataFrame(rows)
    return df_dec


def _run_val_inference(checkpoint_path, data_root, seed, fold, n_splits, batch_size):
    """Run inference on validation set and return (preds, targets, genders)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone_name", "convnext_tiny.fb_in22k_ft_in1k")
    predict_gender = ckpt.get("predict_gender", False)

    model = FaceOcclusionModel(backbone_name=backbone_name, pretrained=False, predict_gender=predict_gender)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    image_dir = _image_dir(data_root)
    df_full, _ = load_dataframes(data_root=data_root)

    if fold is not None:
        splits = get_kfold_splits(df_full, n_splits=n_splits, seed=seed)
        _, val_idx = splits[fold]
        df_val = df_full.iloc[val_idx]
    else:
        _, df_val = split_train_val(df_full, seed=seed)

    val_ds = FaceOcclusionDataset(df_val, image_dir=image_dir, transform=get_val_transforms())
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    all_preds, all_targets, all_genders = [], [], []
    with torch.no_grad():
        for imgs, targets, genders in tqdm(val_loader, desc="Val inference"):
            imgs = imgs.to(device)
            if predict_gender:
                occ, _ = model(imgs)
            else:
                occ = model(imgs)
            all_preds.append(occ.cpu().numpy())
            all_targets.append(targets.numpy())
            all_genders.append(genders.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_genders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostics: error decomposition by gender")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to evaluate on val set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="docs")
    args = parser.parse_args()

    data_root = args.data_root or DEFAULT_DATA_ROOT
    df_full, _ = load_dataframes(data_root=data_root)

    # 1. Distribution
    _distribution_by_gender(df_full, output_dir=args.output_dir)

    # 2. Error decomposition (if checkpoint provided)
    if args.checkpoint:
        print(f"\n=== Error decomposition for {args.checkpoint} ===")
        preds, targets, genders = _run_val_inference(
            args.checkpoint, data_root, args.seed, args.fold, args.n_splits, args.batch_size
        )
        score, err_f, err_m = compute_score(preds, targets, genders)
        print(f"Overall: score={score:.6f} err_f={err_f:.6f} err_m={err_m:.6f} gap={abs(err_f-err_m):.6f}")

        df_dec = _error_decomposition(preds, targets, genders)
        print("\nError by gender × occlusion bin:")
        print(df_dec.to_string(index=False))

        # Save plot
        os.makedirs(args.output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        for gender_label, color in [("F", "#e74c3c"), ("M", "#3498db")]:
            sub = df_dec[df_dec["gender"] == gender_label]
            ax.bar(sub["bin"] + (0.2 if gender_label == "M" else -0.2),
                   sub["weighted_err"], width=0.35, color=color, alpha=0.75,
                   label=gender_label, edgecolor="black")
        ax.set_xlabel("Occlusion bin (low → high)")
        ax.set_ylabel("Weighted error")
        ax.set_title("Error decomposition by gender × occlusion bin")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(args.output_dir, "error_decomposition.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # 3. Correlation male_factor vs gap (if multiple histories available)
    history_files = sorted(glob.glob("data/submissions/checkpoints/**/history*.json", recursive=True))
    if len(history_files) > 1:
        print(f"\n=== Male factor correlation ({len(history_files)} histories) ===")
        rows = []
        for hf in history_files:
            h = json.load(open(hf))
            best_idx = int(np.argmin(h["val_score"]))
            # Try to detect male_factor from directory name
            dirname = os.path.dirname(hf)
            mf = None
            if "mf_" in dirname:
                try:
                    mf = float(dirname.split("mf_")[-1].split("/")[0])
                except ValueError:
                    pass
            rows.append({
                "path": hf,
                "male_factor": mf,
                "score": h["val_score"][best_idx],
                "err_f": h["err_f"][best_idx],
                "err_m": h["err_m"][best_idx],
                "gap": abs(h["err_f"][best_idx] - h["err_m"][best_idx]),
            })
        df_mf = pd.DataFrame(rows)
        if df_mf["male_factor"].notna().any():
            df_mf_valid = df_mf[df_mf["male_factor"].notna()].sort_values("male_factor")
            print(df_mf_valid[["male_factor", "score", "err_f", "err_m", "gap"]].to_string(index=False))
