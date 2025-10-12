# my_data_loader.py
"""
Simple dataset utility for your JAX backprop trainer.

Features:
- MNIST downloader (from public HTTPS mirrors), stored as npz.
- Synthetic "blobs" fallback (10 classes, 784-D).
- get_data(split, index, dir="data") -> (x, y)   # exactly your expected API
- size(split, dir="data") -> int
- prepare_data(dataset="mnist" | "blobs", dir="data", ...) to set things up.

Usage (CLI):
    python my_data_loader.py --dataset mnist --dir data
    python my_data_loader.py --dataset blobs --dir data

Programmatic:
    from my_data_loader import prepare_data, get_data, size
    prepare_data(dataset="mnist", dir="data")     # or "blobs"
    n_train = size("train", "data")
    x0, y0 = get_data("train", 0, "data")
"""

from __future__ import annotations
import argparse
import gzip
import os
import struct
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
from urllib.request import urlopen, Request


# ---------------------------
# Globals & simple cache
# ---------------------------

_ACTIVE_DATASET: Optional[str] = None
_CACHE: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# MNIST download & parse
# ---------------------------

_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Prefer HTTPS mirrors; try in order silently.
_MNIST_BASES = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    # Fallback (HTTP): "http://yann.lecun.com/exdb/mnist/",
]


def _download(url: str, dest: Path) -> None:
    dest_tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": "python-urllib/3"})
    with urlopen(req) as r, open(dest_tmp, "wb") as f:
        f.write(r.read())
    dest_tmp.replace(dest)


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad IDX image magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols)  # (N, 784)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad IDX label magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n)  # (N,)


def _prepare_mnist(dir: str) -> None:
    root = Path(dir) / "mnist"
    _ensure_dir(root)
    # If already prepared, skip
    out_train = root / "train.npz"
    out_test = root / "test.npz"
    if out_train.exists() and out_test.exists():
        return

    # Download .gz files if missing
    local_paths = {k: root / v for k, v in _MNIST_FILES.items()}
    for key, p in local_paths.items():
        if not p.exists():
            filename = _MNIST_FILES[key]
            last_err = None
            for base in _MNIST_BASES:
                try:
                    _download(base + filename, p)
                    break
                except Exception as e:
                    last_err = e
                    continue
            if not p.exists():
                raise RuntimeError(f"Failed to download {filename}: {last_err}")

    # Parse
    train_images = _read_idx_images_gz(local_paths["train_images"])
    train_labels = _read_idx_labels_gz(local_paths["train_labels"])
    test_images = _read_idx_images_gz(local_paths["test_images"])
    test_labels = _read_idx_labels_gz(local_paths["test_labels"])

    # Store compactly (uint8). We'll normalize in get_data().
    np.savez_compressed(out_train, images=train_images.astype(np.uint8), labels=train_labels.astype(np.uint8))
    np.savez_compressed(out_test, images=test_images.astype(np.uint8), labels=test_labels.astype(np.uint8))


# ---------------------------
# Synthetic "blobs" dataset
# ---------------------------

def _prepare_blobs(
    dir: str,
    *,
    classes: int = 10,
    dim: int = 784,
    train_size: int = 60_000,
    test_size: int = 10_000,
    seed: int = 0,
    spread: float = 0.7,
) -> None:
    root = Path(dir) / "blobs"
    _ensure_dir(root)
    out_train = root / "train.npz"
    out_test = root / "test.npz"
    if out_train.exists() and out_test.exists():
        return

    rng = np.random.default_rng(seed)
    # Random class centroids on unit sphere scaled a bit
    centroids = rng.normal(size=(classes, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    centroids *= 2.0  # separate classes a bit

    def _make_split(n: int):
        ys = rng.integers(0, classes, size=n, dtype=np.int32)
        noise = rng.normal(size=(n, dim)).astype(np.float32) * spread
        xs = centroids[ys] + noise
        # Shift/scale to roughly [0,1] just to look similar to MNIST intensity range
        xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
        # Store as uint8 to keep filesize friendly; normalize back in get_data()
        xs_uint8 = np.clip(np.round(xs * 255.0), 0, 255).astype(np.uint8)
        ys_uint8 = ys.astype(np.uint8)
        return xs_uint8, ys_uint8

    Xtr, Ytr = _make_split(train_size)
    Xte, Yte = _make_split(test_size)
    np.savez_compressed(out_train, images=Xtr, labels=Ytr)
    np.savez_compressed(out_test, images=Xte, labels=Yte)


# ---------------------------
# Public API
# ---------------------------

def prepare_data(
    *,
    dataset: str = "mnist",
    dir: str = "data",
    seed: int = 0,
    blobs_classes: int = 10,
    blobs_dim: int = 784,
    blobs_train_size: int = 60_000,
    blobs_test_size: int = 10_000,
    blobs_spread: float = 0.7,
) -> None:
    """
    Prepare (download or generate) the dataset on disk.
    Sets the active dataset for get_data/size().

    dataset: "mnist" or "blobs"
    """
    global _ACTIVE_DATASET
    dataset = dataset.lower()
    if dataset == "mnist":
        _prepare_mnist(dir)
    elif dataset == "blobs":
        _prepare_blobs(
            dir,
            classes=blobs_classes,
            dim=blobs_dim,
            train_size=blobs_train_size,
            test_size=blobs_test_size,
            seed=seed,
            spread=blobs_spread,
        )
    else:
        raise ValueError("dataset must be 'mnist' or 'blobs'")
    _ACTIVE_DATASET = dataset


def _load_split(dataset: str, split: str, dir: str) -> Dict[str, np.ndarray]:
    key = (dataset, split, dir)
    if key in _CACHE:
        return _CACHE[key]

    root = Path(dir) / dataset
    path = root / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{split}' for dataset '{dataset}' not found at {path}. "
            f"Call prepare_data(dataset='{dataset}', dir='{dir}') first."
        )
    arrays = np.load(path, allow_pickle=False, mmap_mode="r")
    # Keep mmap views; convert to arrays on the fly in get_data()
    _CACHE[key] = {"images": arrays["images"], "labels": arrays["labels"]}
    return _CACHE[key]


def _auto_dataset(dir: str) -> str:
    """Pick the active dataset if user didn't call prepare_data()."""
    global _ACTIVE_DATASET
    if _ACTIVE_DATASET is not None:
        return _ACTIVE_DATASET
    if (Path(dir) / "mnist" / "train.npz").exists():
        _ACTIVE_DATASET = "mnist"
    elif (Path(dir) / "blobs" / "train.npz").exists():
        _ACTIVE_DATASET = "blobs"
    else:
        raise RuntimeError(
            "No prepared dataset found. Call prepare_data(dataset='mnist' or 'blobs')."
        )
    return _ACTIVE_DATASET


def get_data(split: str, index: int, dir: str = "data") -> Tuple[np.ndarray, int]:
    """
    Required classroom API:
        returns (1D float32 array, int label)

    - For MNIST: values in [0,1], shape (784,)
    - For blobs: values in [0,1], shape (784,) by default
    """
    dataset = _auto_dataset(dir)
    arrays = _load_split(dataset, split, dir)

    n = int(arrays["labels"].shape[0])
    if index < 0 or index >= n:
        raise IndexError(f"index {index} out of range for split '{split}' (size {n})")

    x_uint8 = np.asarray(arrays["images"][index])
    y_uint8 = int(arrays["labels"][index])

    # Convert to float32 and normalize to [0,1]
    x = (x_uint8.astype(np.float32) / 255.0).reshape(-1)
    return x, y_uint8


def size(split: str, dir: str = "data") -> int:
    """Number of items in a split."""
    dataset = _auto_dataset(dir)
    arrays = _load_split(dataset, split, dir)
    return int(arrays["labels"].shape[0])


# ---------------------------
# CLI
# ---------------------------

def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "blobs"], default="mnist")
    p.add_argument("--dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blobs-classes", type=int, default=10)
    p.add_argument("--blobs-dim", type=int, default=784)
    p.add_argument("--blobs-train-size", type=int, default=60000)
    p.add_argument("--blobs-test-size", type=int, default=10000)
    p.add_argument("--blobs-spread", type=float, default=0.7)
    args = p.parse_args(argv)

    prepare_data(
        dataset=args.dataset,
        dir=args.dir,
        seed=args.seed,
        blobs_classes=args.blobs_classes,
        blobs_dim=args.blobs_dim,
        blobs_train_size=args.blobs_train_size,
        blobs_test_size=args.blobs_test_size,
        blobs_spread=args.blobs_spread,
    )
    print(f"Prepared dataset '{args.dataset}' in {args.dir}")
    print(f"train size: {size('train', args.dir)}")
    print(f"test  size: {size('test', args.dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

