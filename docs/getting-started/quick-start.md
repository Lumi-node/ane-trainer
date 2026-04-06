# 🚀 ANE Trainer Quick Start Guide

Welcome to `ane_trainer`! This package provides a streamlined command-line interface to train small neural networks specifically targeting Apple Neural Engine (ANE) hardware.

This guide will get you up and running quickly.

## Prerequisites

Before you begin, ensure you have the necessary environment set up. Since this package interacts with low-level hardware APIs, specific dependencies (like PyTorch/TensorFlow compiled for Metal/ANE) are assumed to be installed.

```bash
# Assuming you are using pip to install the package
pip install ane_trainer
```

## Core Concepts

The `ane_trainer` package is structured around orchestrating the training pipeline:

1.  **Data Handling (`ane_trainer.data`):** Loading and preprocessing datasets (e.g., MNIST).
2.  **Model Definition (`ane_trainer.models`):** Defining the neural network architecture.
3.  **Training Logic (`ane_trainer.core`):** The engine that manages the training loop, interfacing with ANE APIs.
4.  **Command Line Interface (`ane_trainer.cli`):** The entry point for running training jobs from the terminal.

## 🛠️ Usage Examples

You can use `ane_trainer` in two primary ways: via the **Command Line Interface (CLI)** for quick runs, or by importing modules directly in a **Python script** for custom workflows.

---

### Example 1: Quick CLI Training Run (MNIST)

This is the fastest way to test the package. It uses the default settings to train a basic model on the MNIST dataset.

**Command:**

```bash
ane_trainer train --dataset mnist --epochs 5 --batch-size 32
```

**What happens:**
1.  `ane_trainer/cli.py` parses the arguments.
2.  `ane_trainer/data.py.load_dataset('mnist')` fetches the data.
3.  `ane_trainer/models.py.build_model()` creates the network.
4.  `ane_trainer/core.py.run_training()` executes the loop, leveraging ANE optimizations.

---

### Example 2: Custom Training Script (Using Python API)

If you need fine-grained control over hyperparameters or data loading, import the core components directly into your script (e.g., `custom_train.py`).

**`custom_train.py`:**

```python
import ane_trainer.data as data_loader
import ane_trainer.models as model_builder
import ane_trainer.core as trainer_core

# 1. Load Data
print("Loading custom dataset...")
train_loader, test_loader = data_loader.load_dataset(
    dataset_name="mnist", 
    split="train_test", 
    batch_size=64
)

# 2. Build Model
print("Building model architecture...")
model = model_builder.build_model(layers=[32, 16]) # Example: 2 hidden layers

# 3. Run Training
print("Starting ANE training process...")
results = trainer_core.run_training(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=10,
    learning_rate=0.001
)

print("\n--- Training Complete ---")
print(f"Final Test Accuracy: {results['accuracy']:.4f}")
```

**Execution:**

```bash
python custom_train.py
```

---

### Example 3: Model Inspection and Export

You can use the package utilities to inspect the model structure or prepare it for deployment without running a full training cycle.

**`inspect_model.py`:**

```python
import ane_trainer.models as model_builder
import ane_trainer.utils as utils

# Define a simple model structure
my_model = model_builder.build_model(layers=[128, 64])

# Inspect the structure
print("--- Model Summary ---")
model_builder.print_model_summary(my_model)

# Export the model weights in ANE-compatible format (if supported by the backend)
output_path = "ane_optimized_model.ane"
utils.export_model(my_model, output_path)
print(f"\nModel successfully exported to {output_path}")
```

**Execution:**

```bash
python inspect_model.py
```

## 📚 Module Reference

| Module | Primary Function | Description |
| :--- | :--- | :--- |
| `ane_trainer.cli` | `main()` | Entry point for command-line execution. |
| `ane_trainer.data` | `load_dataset()` | Handles fetching and preparing MNIST/other datasets. |
| `ane_trainer.models` | `build_model()` | Constructs the neural network architecture. |
| `ane_trainer.core` | `run_training()` | Orchestrates the entire training loop on ANE hardware. |
| `ane_trainer.utils` | `export_model()` | Tools for saving and optimizing the trained model. |