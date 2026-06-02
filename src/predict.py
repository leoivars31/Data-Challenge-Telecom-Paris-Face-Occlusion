import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src.dataset import (
    FaceOcclusionDataset, load_dataframes, get_val_transforms, get_tta_transforms,
    DEFAULT_DATA_ROOT, _image_dir,
)
from src.model import FaceOcclusionModel


def predict(
    checkpoint_path="data/submissions/checkpoints/best_model.pt",
    output_path="data/submissions/test_predictions.csv",
    use_tta=True,
    batch_size=64,
    num_workers=4,
    data_root=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone_name", "convnext_tiny.fb_in22k_ft_in1k")

    model = FaceOcclusionModel(backbone_name=backbone_name, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (score={ckpt['best_score']:.6f})")

    # Load test data
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    image_dir = _image_dir(data_root)
    _, df_test = load_dataframes(data_root=data_root)

    if use_tta:
        transform_orig, transform_flip = get_tta_transforms()
        ds_orig = FaceOcclusionDataset(df_test, image_dir=image_dir, transform=transform_orig, is_test=True)
        ds_flip = FaceOcclusionDataset(df_test, image_dir=image_dir, transform=transform_flip, is_test=True)

        loader_orig = torch.utils.data.DataLoader(
            ds_orig, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        loader_flip = torch.utils.data.DataLoader(
            ds_flip, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        preds_orig = _run_inference(model, loader_orig, device)
        preds_flip = _run_inference(model, loader_flip, device)
        predictions = (preds_orig + preds_flip) / 2.0
    else:
        ds = FaceOcclusionDataset(df_test, image_dir=image_dir, transform=get_val_transforms(), is_test=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        predictions = _run_inference(model, loader, device)

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
def _run_inference(model, loader, device):
    all_preds = []
    for imgs, filenames in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission predictions")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data root containing test_students.csv and Crop_224_5fp_100K/")
    parser.add_argument("--checkpoint", type=str, default="data/submissions/checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="data/submissions/test_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    args = parser.parse_args()

    predict(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_tta=not args.no_tta,
        batch_size=args.batch_size,
        data_root=args.data_root,
    )
