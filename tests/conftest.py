"""Pytest fixtures for ane_trainer tests."""

import pytest
import numpy as np
import torch

from ane_trainer.models import build_model


@pytest.fixture
def toy_model():
    """
    Minimal 2-layer model for fast testing.

    Returns a model with standard MNIST architecture:
    - input_size=784
    - hidden_size=128
    - output_size=10
    """
    return build_model(784, 128, 10)


@pytest.fixture
def toy_data():
    """
    Small batch of dummy MNIST data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) where
            - X: shape (64, 784), dtype float32, values in [0, 1]
            - y: shape (64,), dtype int64, values in [0, 9]
    """
    X = np.random.randn(64, 784).astype(np.float32)
    X = np.clip(X, 0, 1)  # Normalize to [0, 1]
    y = np.random.randint(0, 10, 64).astype(np.int64)
    return X, y


@pytest.fixture
def optimizer(toy_model):
    """SGD optimizer for the toy model."""
    return torch.optim.SGD(toy_model.parameters(), lr=0.01)


@pytest.fixture
def loss_fn():
    """Cross-entropy loss function."""
    return torch.nn.CrossEntropyLoss()
