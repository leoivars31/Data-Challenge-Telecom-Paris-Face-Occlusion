import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit


DEFAULT_DATA_ROOT = "data/raw"


def _image_dir(data_root):
    return os.path.join(data_root, "Crop_224_5fp_100K")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms():
    """Returns (original_transform, flip_transform) for TTA."""
    base = get_val_transforms()
    flip = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return base, flip


class FaceOcclusionDataset(Dataset):
    def __init__(self, df, image_dir=None, transform=None, is_test=False):
        if image_dir is None:
            image_dir = _image_dir(DEFAULT_DATA_ROOT)
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, filename
        else:
            occlusion = np.float32(row["FaceOcclusion"])
            gender = np.float32(row["gender"])
            return img, occlusion, gender


def load_dataframes(train_csv=None, test_csv=None, data_root=None):
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    if train_csv is None:
        train_csv = os.path.join(data_root, "train.csv")
    if test_csv is None:
        test_csv = os.path.join(data_root, "test_students.csv")

    df_train = pd.read_csv(train_csv).dropna()
    df_test = pd.read_csv(test_csv).dropna()
    return df_train, df_test


def split_train_val(df, val_ratio=0.15, seed=42):
    """Stratified split by gender + occlusion bins."""
    df = df.copy()
    # Create stratification key from gender and occlusion quantile bins
    df["occ_bin"] = pd.qcut(df["FaceOcclusion"], q=10, labels=False, duplicates="drop")
    df["strat_key"] = df["gender"].astype(str) + "_" + df["occ_bin"].astype(str)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df["strat_key"]))

    df_train = df.iloc[train_idx].drop(columns=["occ_bin", "strat_key"])
    df_val = df.iloc[val_idx].drop(columns=["occ_bin", "strat_key"])
    return df_train, df_val


def get_dataloaders(batch_size=32, val_ratio=0.15, num_workers=4, seed=42, data_root=None):
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    image_dir = _image_dir(data_root)
    df_full, df_test = load_dataframes(data_root=data_root)
    df_train, df_val = split_train_val(df_full, val_ratio=val_ratio, seed=seed)

    train_ds = FaceOcclusionDataset(df_train, image_dir=image_dir, transform=get_train_transforms())
    val_ds = FaceOcclusionDataset(df_val, image_dir=image_dir, transform=get_val_transforms())
    test_ds = FaceOcclusionDataset(df_test, image_dir=image_dir, transform=get_val_transforms(), is_test=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, df_test
