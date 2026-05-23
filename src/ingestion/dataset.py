import os
from typing import Optional, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WaterDataset(Dataset):
    """Loads image-mask pairs, applies transforms, resizes to 256x256."""

    def __init__(self, img_dir: str, mask_dir: str, transform: Optional[A.Compose] = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 200).astype(np.float32)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        return img, mask.unsqueeze(0).float()


def get_transforms(train: bool = True) -> A.Compose:
    """Training: flip/rotate/color jitter + normalize. Val: normalize only."""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_loaders(img_dir: str, mask_dir: str, val_split: float = 0.15,
                test_split: float = 0.15, batch_size: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train/val/test DataLoaders with deterministic random split."""
    full_ds = WaterDataset(img_dir, mask_dir, transform=get_transforms(train=False))
    n = len(full_ds)
    val_n = int(n * val_split)
    test_n = int(n * test_split)
    train_n = n - val_n - test_n
    torch.manual_seed(42)
    indices = torch.randperm(n)

    train_ds = WaterDataset(img_dir, mask_dir, transform=get_transforms(train=True))
    train_ds = torch.utils.data.Subset(train_ds, indices[:train_n])
    val_ds = torch.utils.data.Subset(full_ds, indices[train_n:train_n + val_n])
    test_ds = torch.utils.data.Subset(full_ds, indices[train_n + val_n:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def tile_image(image: np.ndarray, tile_size: int = 256, overlap: int = 32
               ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Split large image into overlapping tiles for inference."""
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles, coords = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            y1, x1 = max(y2 - tile_size, 0), max(x2 - tile_size, 0)
            tile = image[y1:y2, x1:x2]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size))
            tiles.append(tile)
            coords.append((y1, x1, y2, x2))
    return tiles, coords


def merge_tiles(tile_masks: List[np.ndarray], coords: List[Tuple[int, int, int, int]],
                orig_h: int, orig_w: int) -> np.ndarray:
    """Blend overlapping tile predictions into a full-resolution mask."""
    canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
    weights = np.zeros((orig_h, orig_w), dtype=np.float32)
    for mask, (y1, x1, y2, x2) in zip(tile_masks, coords):
        m = cv2.resize(mask, (x2 - x1, y2 - y1))
        canvas[y1:y2, x1:x2] += m
        weights[y1:y2, x1:x2] += 1
    weights = np.maximum(weights, 1)
    return (canvas / weights > 0.5).astype(np.uint8)
