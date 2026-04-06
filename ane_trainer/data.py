"""MNIST dataset loading and caching module."""

import os
from typing import Tuple

import numpy as np
import torchvision.datasets


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset from local path or download if missing.

    Args:
        dataset_path (str): Directory path where MNIST data is/will be stored.
                           Created if missing.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: shape (60000, 28, 28), dtype float32, values in [0.0, 1.0]
            - y_train: shape (60000,), dtype int64, values in range [0, 9]
            - X_test: shape (10000, 28, 28), dtype float32, values in [0.0, 1.0]
            - y_test: shape (10000,), dtype int64, values in range [0, 9]

    Raises:
        OSError: If dataset cannot be created/written (invalid path, no disk space)
        RuntimeError: If download fails and offline cache unavailable

    Normalization:
        - Input images from torchvision are uint8 [0, 255]
        - Divide by 255.0 to normalize to float32 [0.0, 1.0]
        - Labels copied as-is from torchvision (already 0-9)

    Caching:
        - On first call with empty dataset_path: Downloads MNIST via torchvision
        - Subsequent calls: Uses cached files from dataset_path
        - No re-download if files already present
    """
    # Validate that parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(dataset_path))
    if parent_dir and not os.path.isdir(parent_dir):
        raise OSError(f"Dataset path parent must exist: {parent_dir}")

    # Create dataset_path directory if it doesn't exist
    try:
        os.makedirs(dataset_path, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create dataset directory: {dataset_path}") from e

    try:
        # Load training set
        mnist_train = torchvision.datasets.MNIST(
            root=dataset_path, train=True, download=True
        )
        # Load test set
        mnist_test = torchvision.datasets.MNIST(
            root=dataset_path, train=False, download=True
        )
    except Exception as e:
        raise RuntimeError("Failed to download MNIST") from e

    # Extract and normalize training data
    X_train = np.array([np.array(img) for img, _ in mnist_train], dtype=np.float32)
    y_train = np.array([label for _, label in mnist_train], dtype=np.int64)

    # Extract and normalize test data
    X_test = np.array([np.array(img) for img, _ in mnist_test], dtype=np.float32)
    y_test = np.array([label for _, label in mnist_test], dtype=np.int64)

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test
