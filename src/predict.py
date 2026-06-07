import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as T

from src.dataset import (
    FaceOcclusionDataset, load_dataframes, get_val_transforms, get_tta_transforms,
    DEFAULT_DATA_ROOT, _image_dir, IMAGENET_MEAN, IMAGENET_STD,
)
from src.model import FaceOcclusionModel


def _make_scale_transforms(scale):
    """Create (orig, flip) transforms for a given scale factor."""
    size = int(224 * scale)
    orig = T.Compose([
        T.Resize((size, size)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    flip = T.Compose([
        T.Resize((size, size)),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return orig, flip


def predict(
    checkpoint_path="data/submissions/checkpoints/best_model.pt",
    output_path="data/submissions/test_predictions.csv",
    use_tta=True,
    batch_size=64,
    num_workers=4,
    data_root=None,
    checkpoint_paths=None,
    tta_scales=None,
    calibrators_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve checkpoint list
    if checkpoint_paths is None:
        checkpoint_paths = [checkpoint_path]

    if tta_scales is None:
        tta_scales = [1.0]

    # Load test data
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    image_dir = _image_dir(data_root)
    _, df_test = load_dataframes(data_root=data_root)

    # Accumulate predictions across models
    all_model_preds = []
    all_gender_logits = []

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone_name = ckpt.get("backbone_name", "convnext_tiny.fb_in22k_ft_in1k")
        predict_gender = ckpt.get("predict_gender", False)

        model = FaceOcclusionModel(
            backbone_name=backbone_name, pretrained=False, predict_gender=predict_gender
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, score={ckpt['best_score']:.6f})")

        # TTA across scales
        scale_preds = []
        scale_gender_logits = []
        for scale in tta_scales:
            if scale == 1.0 and use_tta:
                transform_orig, transform_flip = get_tta_transforms()
            elif scale == 1.0 and not use_tta:
                transform_orig = get_val_transforms()
                transform_flip = None
            else:
                transform_orig, transform_flip = _make_scale_transforms(scale)

            ds_orig = FaceOcclusionDataset(df_test, image_dir=image_dir, transform=transform_orig, is_test=True)
            loader_orig = torch.utils.data.DataLoader(
                ds_orig, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
            preds_orig, glogits_orig = _run_inference(model, loader_orig, device, predict_gender)
            scale_preds.append(preds_orig)
            if predict_gender:
                scale_gender_logits.append(glogits_orig)

            if transform_flip is not None:
                ds_flip = FaceOcclusionDataset(df_test, image_dir=image_dir, transform=transform_flip, is_test=True)
                loader_flip = torch.utils.data.DataLoader(
                    ds_flip, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )
                preds_flip, glogits_flip = _run_inference(model, loader_flip, device, predict_gender)
                scale_preds.append(preds_flip)
                if predict_gender:
                    scale_gender_logits.append(glogits_flip)

        model_pred = np.mean(scale_preds, axis=0)
        all_model_preds.append(model_pred)

        if predict_gender and scale_gender_logits:
            all_gender_logits.append(np.mean(scale_gender_logits, axis=0))

    # Ensemble average
    predictions = np.mean(all_model_preds, axis=0)

    # Calibration (if available)
    if calibrators_path and all_gender_logits:
        from src.calibration import load_calibrators, apply_calibration
        calibrators = load_calibrators(calibrators_path)
        avg_gender_logits = np.mean(all_gender_logits, axis=0)
        pred_genders = (1.0 / (1.0 + np.exp(-avg_gender_logits))) > 0.5
        predictions = apply_calibration(predictions, pred_genders, calibrators)

    # Clip to [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)

    # Build submission
    submission = pd.DataFrame({
        "filename": df_test["filename"].values,
        "FaceOcclusion": predictions,
        "gender": "x",
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")


@torch.no_grad()
def _run_inference(model, loader, device, predict_gender=False):
    all_preds = []
    all_gender_logits = []
    for imgs, filenames in tqdm(loader, desc="Predicting", leave=False):
        imgs = imgs.to(device)
        if predict_gender:
            occ, glogit = model(imgs)
            all_preds.append(occ.cpu().numpy())
            all_gender_logits.append(glogit.cpu().numpy())
        else:
            preds = model(imgs)
            all_preds.append(preds.cpu().numpy())
    preds_out = np.concatenate(all_preds)
    glogits_out = np.concatenate(all_gender_logits) if all_gender_logits else None
    return preds_out, glogits_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission predictions")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data root containing test_students.csv and Crop_224_5fp_100K/")
    parser.add_argument("--checkpoint", type=str, default="data/submissions/checkpoints/best_model.pt",
                        help="Single checkpoint path (used if --checkpoints not given)")
    parser.add_argument("--checkpoints", nargs="+", type=str, default=None,
                        help="Multiple checkpoint paths for ensemble")
    parser.add_argument("--output", type=str, default="data/submissions/test_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--tta_scales", nargs="+", type=float, default=[1.0],
                        help="TTA scales (e.g., 1.0 0.9 1.1)")
    parser.add_argument("--calibrators", type=str, default=None,
                        help="Path to calibrators pickle (for gender-conditional calibration)")
    args = parser.parse_args()

    predict(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_tta=not args.no_tta,
        batch_size=args.batch_size,
        data_root=args.data_root,
        checkpoint_paths=args.checkpoints,
        tta_scales=args.tta_scales,
        calibrators_path=args.calibrators,
    )
