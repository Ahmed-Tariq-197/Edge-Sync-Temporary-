#!/usr/bin/env python3
"""Train the MobileNetV2 Re-ID embedder with an identity-classification loss.

Example:
  python tools/ablation/task6/train_mobilenetv2_reid.py \
      --train-dir data/market1501/bounding_box_train --output checkpoints/mobilenetv2_reid.pt
Only training subjects are used; test subjects must never appear in --train-dir.
"""
import argparse
import random
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mobilenetv2_reid_model import (
    IMAGENET_MEAN, IMAGENET_STD, INPUT_HEIGHT, INPUT_WIDTH, ReidTrainingModel,
)
from edgesync_identity.ablation.market_dataset import ImageRecord, list_records

DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3.5e-4
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_SEED = 0
DEFAULT_NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1
RANDOM_ERASING_PROBABILITY = 0.5
CROP_PADDING_PIXELS = 10
LOG_INTERVAL_EPOCHS = 1
JUNK_IDENTITY = -1


class ReidImageDataset(Dataset):
    def __init__(self, records: List[ImageRecord], label_of: dict, transform):
        self.records, self.label_of, self.transform = records, label_of, transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), self.label_of[record.person_id]


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(CROP_PADDING_PIXELS),
        transforms.RandomCrop((INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=RANDOM_ERASING_PROBABILITY),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Re-ID embedder.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = [r for r in list_records(args.train_dir) if r.person_id != JUNK_IDENTITY]
    identities = sorted({r.person_id for r in records})
    label_of = {pid: index for index, pid in enumerate(identities)}
    loader = DataLoader(ReidImageDataset(records, label_of, build_transform()),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers)

    model = ReidTrainingModel(num_identities=len(identities)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        if epoch % LOG_INTERVAL_EPOCHS == 0:
            print(f"epoch {epoch}/{args.epochs} loss {total_loss / max(batches, 1):.4f}")

    torch.save({"embedder": model.embedder.state_dict(), "num_identities": len(identities)}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
