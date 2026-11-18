"""Data loading and fixed partitions for the ECS708 Fashion-MNIST coursework (Part 0).

All experiments import from here so that every model sees exactly the same
images, preprocessing and partitions (rules R1, R2, R4).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fashion"
SPLITS_DIR = ROOT / "splits"
SEED = 708  # the single fixed seed used for every partition

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


def _ensure_downloaded() -> None:
    """Download the official files with torchvision if they are not already in data/fashion."""
    if all((DATA_DIR / name).exists() for name in FILES.values()):
        return
    from torchvision.datasets import FashionMNIST  # only needed for the download

    tmp = ROOT / "data" / "_torchvision"
    for train in (True, False):
        FashionMNIST(tmp, train=train, download=True)
    raw = tmp / "FashionMNIST" / "raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        (DATA_DIR / name).write_bytes((raw / name).read_bytes())


def load_official():
    """Return the official (X_train, y_train, X_test, y_test); images are uint8 28x28."""
    _ensure_downloaded()
    return (
        _read_images(DATA_DIR / FILES["train_images"]),
        _read_labels(DATA_DIR / FILES["train_labels"]),
        _read_images(DATA_DIR / FILES["test_images"]),
        _read_labels(DATA_DIR / FILES["test_labels"]),
    )


def load_splits() -> dict[str, np.ndarray]:
    """Saved index arrays produced by scripts/make_splits.py."""
    path = SPLITS_DIR / f"splits_seed{SEED}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/make_splits.py` first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def get_partitions(cpu_subset: bool = False):
    """Return dict of (X, y) for 'train', 'val', 'test'.

    X is float32 in [0, 1], shape (N, 784). Pixels are only divided by 255,
    a fixed transform that uses no statistics, so it cannot leak information
    between partitions. Any fitted scaler must be fitted on 'train' only (R2).
    cpu_subset=True swaps the 50k training partition for the fixed stratified 10k subset.
    """
    Xtr, ytr, Xte, yte = load_official()
    s = load_splits()
    tr_idx = s["train_subset10k_idx"] if cpu_subset else s["train_idx"]

    def prep(X):
        return (X.reshape(len(X), -1) / 255.0).astype(np.float32)

    return {
        "train": (prep(Xtr[tr_idx]), ytr[tr_idx]),
        "val": (prep(Xtr[s["val_idx"]]), ytr[s["val_idx"]]),
        "test": (prep(Xte), yte),
    }


def get_cluster_subsets():
    """The fixed ~3k/1k/1k subsets used for every clustering / feature-extraction representation."""
    Xtr, ytr, Xte, yte = load_official()
    s = load_splits()

    def prep(X):
        return (X.reshape(len(X), -1) / 255.0).astype(np.float32)

    return {
        "train": (prep(Xtr[s["cluster_train_idx"]]), ytr[s["cluster_train_idx"]]),
        "val": (prep(Xtr[s["cluster_val_idx"]]), ytr[s["cluster_val_idx"]]),
        "test": (prep(Xte[s["cluster_test_idx"]]), yte[s["cluster_test_idx"]]),
    }


def split_summary() -> dict:
    with open(SPLITS_DIR / f"splits_seed{SEED}.json") as f:
        return json.load(f)
