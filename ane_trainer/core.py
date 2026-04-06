"""Training logic and hardware abstraction layer for ANE-aware forward passes and training steps."""

import numpy as np
import torch

from ane_trainer.utils import is_apple_silicon


def ane_forward_pass(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """
    Execute forward pass with ANE acceleration (if available) or CPU fallback.

    Args:
        model (torch.nn.Module): PyTorch model in eval mode
        x (np.ndarray): Input batch, shape (batch_size, 784), dtype float32

    Returns:
        np.ndarray: Output logits, shape (batch_size, 10), dtype float32

    Hardware Behavior:
        On Apple Silicon (is_apple_silicon() == True):
            - ATTEMPT: Use ANE via reverse-engineered APIs (implementation TBD)
            - FALLBACK: If ANE unavailable, fall back to CPU silently
        On non-Apple hardware (is_apple_silicon() == False):
            - Use CPU execution via PyTorch

        In both cases: Same output dtype/shape, no errors raised

    Error Handling:
        - No exceptions raised even if ANE unavailable
        - Graceful fallback to CPU if ANE fails silently
        - Raises ValueError if input shape/dtype invalid (development safeguard)

    Determinism:
        - In eval mode, given same x produces identical output across calls
        - No random number generation in forward pass

    Data Conversion:
        Input (numpy):  x: (N, 784) float32 [0.0, 1.0]
          ↓
        Convert to PyTorch: torch.from_numpy(x).float() → (N, 784) float32 tensor
          ↓
        Forward: model(x) → (N, 10) float32 tensor
          ↓
        Convert back to numpy: tensor.detach().numpy() → (N, 10) float32 ndarray
          ↓
        Output (numpy): (N, 10) float32
    """
    # Validate input shape and dtype
    if x.ndim != 2 or x.shape[1] != 784:
        raise ValueError(
            f"Expected input shape (batch_size, 784), got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(
            f"Expected input dtype float32, got {x.dtype}"
        )

    # Ensure eval mode (deterministic)
    model.eval()

    # Convert to tensor
    x_tensor = torch.from_numpy(x).float()

    # Forward pass with no gradient computation
    with torch.no_grad():
        # Check if on Apple Silicon hardware
        if is_apple_silicon():
            # FUTURE: Attempt ANE execution here
            # For now: Fall through to CPU
            pass

        # CPU path (always available)
        output = model(x_tensor)

    # Convert back to numpy and ensure float32
    return output.detach().numpy().astype(np.float32)


def train_step(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module
) -> float:
    """
    Perform one training iteration: forward pass, loss computation, backward pass, weight update.

    Args:
        model (torch.nn.Module): PyTorch model in training mode
        x (np.ndarray): Input batch, shape (batch_size, 784), dtype float32
        y (np.ndarray): Target labels, shape (batch_size,), dtype int64, values [0, 9]
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD) with learning rate set
        loss_fn (torch.nn.Module): Loss function (e.g., CrossEntropyLoss)

    Returns:
        float: Scalar loss value (float, > 0 for classification)

    Execution Sequence:
        1. Set model to training mode (model.train())
        2. Zero optimizer gradients (optimizer.zero_grad())
        3. Forward pass: x_tensor = torch.from_numpy(x), logits = model(x_tensor) → (batch_size, 10)
        4. Convert y to tensor: y_tensor = torch.from_numpy(y).long()
        5. Compute loss: loss = loss_fn(logits_tensor, y_tensor)
        6. Backward pass: loss.backward()
        7. Optimizer step: optimizer.step()
        8. Return scalar loss value: float(loss.item())

    Side Effects:
        - Model weights are updated (detached from input state)
        - Optimizer state is updated
        - No state outside model/optimizer is modified

    Error Handling:
        Raises ValueError if:
            - x.shape[0] != y.shape[0] (mismatched batch sizes)
            - y.dtype is not int64 or y contains values outside [0, 9]

        No exception for invalid loss value (NaN/Inf) — propagates to caller
    """
    # Validate batch size match
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} samples"
        )

    # Validate y dtype and values
    if y.dtype != np.int64:
        raise ValueError(
            f"Expected y dtype int64, got {y.dtype}"
        )

    # Validate y contains valid class indices
    if np.any(y < 0) or np.any(y >= 10):
        raise ValueError(
            f"Expected y values in range [0, 9], got values in [{y.min()}, {y.max()}]"
        )

    # Set model to training mode
    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Validate input shape and dtype (same as ane_forward_pass)
    if x.ndim != 2 or x.shape[1] != 784:
        raise ValueError(
            f"Expected input shape (batch_size, 784), got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(
            f"Expected input dtype float32, got {x.dtype}"
        )

    # Convert x to tensor
    x_tensor = torch.from_numpy(x).float()

    # Forward pass (with gradient tracking)
    logits = model(x_tensor)

    # Convert y to tensor
    y_tensor = torch.from_numpy(y).long()

    # Compute loss
    loss = loss_fn(logits, y_tensor)

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Return scalar loss value
    return float(loss.item())
