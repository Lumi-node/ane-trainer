#!/usr/bin/env python3
"""
Example: Train MNIST model using ane_trainer.

Demonstrates complete workflow: load data, build model, train for 3 epochs, save.
Runs without command-line arguments - all configuration is hardcoded.
"""

import os
import sys

import numpy as np
import torch

from ane_trainer.data import load_dataset
from ane_trainer.models import build_model
from ane_trainer.core import train_step


def main():
    """Main function for example script."""
    # Configuration
    dataset_path = "./mnist_data"
    model_path = "./trained_model.pt"
    epochs = 3
    batch_size = 32
    learning_rate = 0.01

    try:
        # Step 1: Load dataset
        print("Loading MNIST dataset...")
        X_train, y_train, X_test, y_test = load_dataset(dataset_path)
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")

        # Step 2: Build model
        print("Building model...")
        model = build_model(input_size=784, hidden_size=128, output_size=10)
        print(f"  Model created: input_size=784, hidden_size=128, output_size=10")

        # Step 3: Initialize optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Step 4: Flatten training data for 784-dimensional input
        X_train_flat = X_train.reshape(-1, 28 * 28).astype(np.float32)

        # Step 5: Training loop
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Iterate through batches
            for batch_start in range(0, len(X_train_flat), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train_flat))
                x_batch = X_train_flat[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                # Perform training step
                loss = train_step(model, x_batch, y_batch, optimizer, loss_fn)
                epoch_loss += loss
                num_batches += 1

            # Compute average loss for epoch
            avg_loss = epoch_loss / num_batches

            # Log epoch loss
            print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}")

        # Step 6: Save model
        print(f"Saving model to {model_path}...")
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path)
        print(f"  Model saved: {model_size} bytes")

        print("Done!")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
