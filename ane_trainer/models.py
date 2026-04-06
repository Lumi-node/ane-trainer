"""
PyTorch neural network model definition.

Responsibility: Define the 2-layer feedforward architecture used for MNIST training.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    2-layer feedforward neural network.

    Architecture:
        1. Linear(input_size, hidden_size)
        2. ReLU()
        3. Linear(hidden_size, output_size)

    No softmax (loss function expects raw logits).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the 2-layer network.

        Args:
            input_size (int): Dimension of input features (e.g., 784 for flattened 28x28)
            hidden_size (int): Dimension of hidden layer (e.g., 128)
            output_size (int): Dimension of output (e.g., 10 for MNIST classes)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_size)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def build_model(input_size: int, hidden_size: int, output_size: int) -> torch.nn.Module:
    """
    Create a 2-layer feedforward neural network.

    Args:
        input_size (int): Dimension of input features (e.g., 784 for flattened 28x28)
        hidden_size (int): Dimension of hidden layer (e.g., 128)
        output_size (int): Dimension of output (e.g., 10 for MNIST classes)

    Returns:
        torch.nn.Module: PyTorch model with forward() method.
                        Output shape: (batch_size, output_size)
                        Output dtype: float32

    Architecture (sequential):
        1. Linear(input_size, hidden_size)
        2. ReLU()
        3. Linear(hidden_size, output_size)

    Parameters:
        - All parameters initialized with default PyTorch initialization
        - All parameters have requires_grad=True (trainable)

    Behavior:
        - In training mode (model.train()): No dropout/batch norm (not used in MVP)
        - In eval mode (model.eval()): Deterministic; same input → same output
    """
    return SimpleNN(input_size, hidden_size, output_size)
