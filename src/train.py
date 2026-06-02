import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm

from src.utils import set_seed, weighted_mse_loss, compute_score
from src.dataset import get_dataloaders
from src.model import FaceOcclusionModel


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None

    for imgs, targets, genders in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(imgs)
            loss = weighted_mse_loss(preds, targets)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, use_amp=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_genders = []
    total_loss = 0.0
    n_batches = 0

    for imgs, targets, genders in tqdm(loader, desc="  Val", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(imgs)
            loss = weighted_mse_loss(preds, targets)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_genders.append(genders)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_genders = torch.cat(all_genders).numpy()

    val_loss = total_loss / n_batches
    score, err_f, err_m = compute_score(all_preds, all_targets, all_genders)
    return val_loss, score, err_f, err_m


def train(
    backbone_name="convnext_tiny.fb_in22k_ft_in1k",
    batch_size=32,
    num_epochs=20,
    freeze_epochs=2,
    lr_head=1e-4,
    lr_backbone=1e-5,
    patience=5,
    seed=42,
    checkpoint_dir="data/submissions/checkpoints",
    data_root=None,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=batch_size, seed=seed, data_root=data_root
    )

    # Model
    model = FaceOcclusionModel(backbone_name=backbone_name).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_score = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_score": [], "err_f": [], "err_m": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Freeze/unfreeze backbone
        if epoch <= freeze_epochs:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                model.head.parameters(), lr=lr_head, weight_decay=1e-2
            )
        elif epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": lr_backbone},
                {"params": model.head.parameters(), "lr": lr_head},
            ], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - freeze_epochs
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_loss, val_score, err_f, err_m = validate(model, val_loader, device, use_amp=use_amp)

        if epoch > freeze_epochs:
            scheduler.step()

        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        print(f"  val_score={val_score:.6f}  err_f={err_f:.6f}  err_m={err_m:.6f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)
        history["err_f"].append(err_f)
        history["err_m"].append(err_m)

        # Save history to JSON after each epoch
        history_path = os.path.join(checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            epochs_no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_score": best_score,
                "backbone_name": backbone_name,
            }, ckpt_path)
            print(f"  -> Saved best model (score={best_score:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best score: {best_score:.6f}")
    return best_score, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face occlusion model")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data root containing train.csv and Crop_224_5fp_100K/")
    parser.add_argument("--backbone", type=str, default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="data/submissions/checkpoints")
    args = parser.parse_args()

    train(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        freeze_epochs=args.freeze_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        patience=args.patience,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
    )
