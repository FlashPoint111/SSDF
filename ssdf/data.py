"""Load prepared face sequences and frozen context vectors for SSDF."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073))[:, None, None]
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711))[:, None, None]


def load_face(path):
    with Image.open(path) as source:
        image = ImageOps.pad(
            source.convert("RGB"), (224, 224),
            method=Image.Resampling.BICUBIC, color=(0, 0, 0),
        )
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return (pixels - CLIP_MEAN) / CLIP_STD


def load_context(path):
    with np.load(path, allow_pickle=False) as cache:
        vector = np.asarray(cache["s_M"], dtype=np.float32).copy()
    if vector.shape != (2048,) or not np.isfinite(vector).all():
        raise ValueError("Expected one finite 2048-dimensional context vector")
    return torch.from_numpy(vector)


class PreparedVideoDataset(Dataset):
    """Read one externally prepared face sequence and its paired Qwen vector."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        paths = [Path(path) for path in record["face_paths"]]
        if len(paths) != 16:
            raise ValueError("Expected one prepared face frame per model position")
        item = {
            "sample_id": str(record["sample_id"]),
            "faces": torch.stack([load_face(path) for path in paths]),
            "context": load_context(record["context_path"]),
        }
        if "label" in record:
            item["label"] = int(record["label"])
        return item


def collate_videos(items):
    batch = {
        "sample_ids": [item["sample_id"] for item in items],
        "faces": torch.stack([item["faces"] for item in items]),
        "context": torch.stack([item["context"] for item in items]),
    }
    labeled = ["label" in item for item in items]
    if any(labeled) and not all(labeled):
        raise ValueError("A batch cannot mix labeled and unlabeled records")
    if all(labeled):
        batch["labels"] = torch.tensor([item["label"] for item in items], dtype=torch.long)
    return batch


def make_dataloader(records, *, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        PreparedVideoDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_videos,
    )
