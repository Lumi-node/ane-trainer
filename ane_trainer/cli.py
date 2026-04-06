"""CLI interface for ane_trainer training orchestration."""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch

from ane_trainer.data import load_dataset
from ane_trainer.models import build_model
from ane_trainer.core import train_step


def main_cli() -> int:
    """
    Parse command-line arguments and invoke main().

    Returns:
        int: Exit code from main() or 2 from argparse if argument parsing fails

    CLI Arguments:
        --dataset PATH (required, str)
            Path to MNIST dataset directory (created if missing)

        --epochs N (required, int)
            Number of training epochs, range [1, 1000]

        --output PATH (required, str)
            Path where to save trained model (.pt file)
            Parent directory must exist; file is created/overwritten

        --batch-size N (optional, int, default 32)
            Training batch size, range [1, 1024]

        --learning-rate LR (optional, float, default 0.01)
            SGD learning rate, range (0.0, 1.0]

    Help Text:
        Shown with --help or -h
        Includes description of each argument and defaults

    Error Handling:
        If argument parsing fails (missing required, invalid type/value):
            - argparse prints usage + error to stderr
            - Exit code 2 (standard argparse behavior)
    """
    parser = argparse.ArgumentParser(
        description="Train neural networks on Apple Neural Engine via reverse-engineered APIs"
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to MNIST dataset directory (created if missing)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where to save trained model (.pt file)"
    )

    # Optional arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate (default: 0.01)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Delegate to main()
    return main(args)


def main(args: argparse.Namespace) -> int:
    """
    Orchestrate training loop: load data, build model, train for N epochs, save weights.

    Args:
        args (argparse.Namespace): Parsed command-line arguments with fields:
            - dataset (str): Path to MNIST dataset directory
            - epochs (int): Number of training epochs
            - output (str): Path where to save .pt model file
            - batch_size (int): Training batch size (default 32)
            - learning_rate (float): SGD learning rate (default 0.01)

    Returns:
        int: Exit code
            - 0 on successful training and model save
            - 1 on any error (dataset load, training, file I/O)

    Execution Flow:
        1. Validate output directory exists:
            - Split output path into dir + filename
            - Check os.path.isdir(output_dir)
            - Exit with 1 if missing

        2. Load dataset:
            - X_train, y_train, X_test, y_test = load_dataset(args.dataset)
            - Catch OSError/RuntimeError, print error to stderr, exit 1

        3. Build model:
            - model = build_model(input_size=784, hidden_size=128, output_size=10)

        4. Initialize optimizer and loss:
            - optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            - loss_fn = torch.nn.CrossEntropyLoss()

        5. Training loop:
            for epoch in range(args.epochs):
                epoch_loss = 0.0
                num_batches = 0

                for batch_start in range(0, len(X_train), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(X_train))
                    x_batch = X_train[batch_start:batch_end]
                    y_batch = y_train[batch_start:batch_end]

                    try:
                        loss = train_step(model, x_batch, y_batch, optimizer, loss_fn)
                        epoch_loss += loss
                        num_batches += 1
                    except Exception as e:
                        print(f"Error during training: {e}", file=sys.stderr)
                        return 1

                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}")

        6. Save model:
            - torch.save(model.state_dict(), args.output)
            - Catch OSError, print error, exit 1

        7. Return 0

    Stdout Logging Format:
        For each epoch N (1-indexed), print exactly:
        "Epoch {N}: Loss {avg_loss:.4f}\n"

        Example for 3 epochs:
        Epoch 1: Loss 2.3015
        Epoch 2: Loss 2.1234
        Epoch 3: Loss 1.9876
    """
    # Step 1: Validate output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    # Step 2: Load dataset
    try:
        X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    except (OSError, RuntimeError) as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    # Flatten the data from (N, 28, 28) to (N, 784)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    # Step 3: Build model
    model = build_model(784, 128, 10)

    # Step 4: Initialize optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Step 5: Training loop
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(X_train), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(X_train))
            x_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            try:
                loss = train_step(model, x_batch, y_batch, optimizer, loss_fn)
                epoch_loss += loss
                num_batches += 1
            except Exception as e:
                print(f"Error during training: {e}", file=sys.stderr)
                return 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}")

    # Step 6: Save model
    try:
        torch.save(model.state_dict(), args.output)
    except OSError as e:
        print(f"Error saving model: {e}", file=sys.stderr)
        return 1

    # Step 7: Return 0 on success
    return 0
