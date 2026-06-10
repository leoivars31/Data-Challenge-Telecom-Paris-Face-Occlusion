import argparse
import json
import math
import os
import torch
import numpy as np
from tqdm import tqdm

from src.utils import (
    set_seed, weighted_mse_loss, fairness_loss, metric_surrogate_loss,
    groupdro_loss, compute_score,
)
from src.dataset import get_dataloaders, get_fold_dataloaders
from src.model import FaceOcclusionModel


def _select_loss(loss_name, preds, targets, genders, male_factor, fairness_lambda):
    """Select and compute the loss based on config."""
    if loss_name == "surrogate":
        return metric_surrogate_loss(preds, targets, genders)
    elif loss_name == "groupdro":
        return groupdro_loss(preds, targets, genders)
    elif fairness_lambda > 0:
        return fairness_loss(preds, targets, genders, fairness_lambda)
    else:
        return weighted_mse_loss(preds, targets, genders=genders, male_factor=male_factor)


def train_one_epoch(model, loader, optimizer, device, scaler=None,
                    fairness_lambda=0.0, male_factor=1.0, loss_name="wmse",
                    predict_gender=False, gender_loss_weight=0.2,
                    grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    bce = torch.nn.BCEWithLogitsLoss() if predict_gender else None

    optimizer.zero_grad()
    for step, (imgs, targets, genders) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        imgs = imgs.to(device)
        targets = targets.to(device)
        genders = genders.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if predict_gender:
                occ_preds, gender_logits = model(imgs)
                loss_occ = _select_loss(loss_name, occ_preds, targets, genders, male_factor, fairness_lambda)
                loss_gender = bce(gender_logits, genders)
                loss = loss_occ + gender_loss_weight * loss_gender
            else:
                preds = model(imgs)
                loss = _select_loss(loss_name, preds, targets, genders, male_factor, fairness_lambda)

        loss = loss / grad_accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, use_amp=False, predict_gender=False):
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
            if predict_gender:
                occ_preds, _ = model(imgs)
            else:
                occ_preds = model(imgs)
            loss = weighted_mse_loss(occ_preds, targets)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(occ_preds.cpu())
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
    num_epochs=30,
    freeze_epochs=2,
    lr_head=3e-4,
    lr_backbone=3e-5,
    patience=7,
    warmup_epochs=2,
    seed=42,
    checkpoint_dir="data/submissions/checkpoints",
    data_root=None,
    fairness_lambda=0.0,
    male_factor=1.0,
    loss_name="wmse",
    occlusion_safe=True,
    fold=None,
    n_splits=5,
    predict_gender=False,
    gender_loss_weight=0.2,
    grad_accum_steps=1,
    val_ratio=0.15,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Log config
    print(f"Loss: {loss_name} | male_factor: {male_factor} | fairness_lambda: {fairness_lambda} | grad_accum: {grad_accum_steps}")
    print(f"occlusion_safe: {occlusion_safe} | predict_gender: {predict_gender}")
    if fold is not None:
        print(f"Fold: {fold}/{n_splits}")

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")

    # Data
    if fold is not None:
        train_loader, val_loader = get_fold_dataloaders(
            fold=fold, n_splits=n_splits, batch_size=batch_size,
            occlusion_safe=occlusion_safe, seed=seed, data_root=data_root,
        )
    else:
        train_loader, val_loader, _, _ = get_dataloaders(
            batch_size=batch_size, seed=seed, data_root=data_root,
            occlusion_safe=occlusion_safe, val_ratio=val_ratio,
        )

    # Model
    model = FaceOcclusionModel(
        backbone_name=backbone_name, predict_gender=predict_gender
    ).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_score = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_score": [], "err_f": [], "err_m": [], "lr": []}

    # File naming
    suffix = f"_fold{fold}" if fold is not None else ""
    ckpt_filename = f"best_model{suffix}.pt"
    history_filename = f"history{suffix}.json"

    # --- Phase 1: freeze backbone, train head only ---
    model.freeze_backbone()
    head_params = list(model.head.parameters())
    if predict_gender:
        head_params += list(model.gender_head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-2)

    # --- Phase 2 optimizer (created after freeze_epochs) ---
    finetune_epochs = num_epochs - freeze_epochs

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Transition: unfreeze backbone and create finetuning optimizer + scheduler
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            param_groups = [
                {"params": model.backbone.parameters(), "lr": lr_backbone},
                {"params": model.head.parameters(), "lr": lr_head},
            ]
            if predict_gender:
                param_groups.append({"params": model.gender_head.parameters(), "lr": lr_head})
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
            # Cosine schedule with linear warmup
            def lr_lambda(current_step, warmup_steps=warmup_epochs, total_steps=finetune_epochs):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler,
            fairness_lambda=fairness_lambda, male_factor=male_factor,
            loss_name=loss_name, predict_gender=predict_gender,
            gender_loss_weight=gender_loss_weight,
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_score, err_f, err_m = validate(
            model, val_loader, device, use_amp=use_amp, predict_gender=predict_gender,
        )

        current_lr = optimizer.param_groups[-1]['lr']
        if epoch > freeze_epochs:
            scheduler.step()

        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.2e}")
        print(f"  val_score={val_score:.6f}  err_f={err_f:.6f}  err_m={err_m:.6f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)
        history["err_f"].append(err_f)
        history["err_m"].append(err_m)
        history["lr"].append(current_lr)

        # Save history to JSON after each epoch
        history_path = os.path.join(checkpoint_dir, history_filename)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            epochs_no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_score": best_score,
                "backbone_name": backbone_name,
                "predict_gender": predict_gender,
                "fold": fold,
                "male_factor": male_factor,
                "loss": loss_name,
                "seed": seed,
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--lr_backbone", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fairness_lambda", type=float, default=0.0,
                        help="(Deprecated) Fairness regularization weight")
    parser.add_argument("--male_factor", type=float, default=1.0,
                        help="Weight multiplier for male samples in loss (>1 = more weight on males)")
    parser.add_argument("--loss", type=str, default="wmse",
                        choices=["wmse", "surrogate", "groupdro"],
                        help="Loss function to use")
    parser.add_argument("--occlusion_safe", action="store_true", default=True,
                        help="Use occlusion-safe augmentations (default)")
    parser.add_argument("--no_occlusion_safe", dest="occlusion_safe", action="store_false",
                        help="Use aggressive augmentations (legacy)")
    parser.add_argument("--fold", type=int, default=None,
                        help="K-fold index (0..n_splits-1). If None, use single 85/15 split")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of folds for K-fold CV")
    parser.add_argument("--predict_gender", action="store_true",
                        help="Enable auxiliary gender prediction head")
    parser.add_argument("--gender_loss_weight", type=float, default=0.2,
                        help="Weight of the gender BCE loss (multi-task)")
    parser.add_argument("--checkpoint_dir", type=str, default="data/submissions/checkpoints")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * steps)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation split ratio when not using k-fold (default: 0.15 = 85/15)")
    args = parser.parse_args()

    train(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        freeze_epochs=args.freeze_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        fairness_lambda=args.fairness_lambda,
        male_factor=args.male_factor,
        loss_name=args.loss,
        occlusion_safe=args.occlusion_safe,
        fold=args.fold,
        n_splits=args.n_splits,
        predict_gender=args.predict_gender,
        gender_loss_weight=args.gender_loss_weight,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        grad_accum_steps=args.grad_accum_steps,
        val_ratio=args.val_ratio,
    )
