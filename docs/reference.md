# ANE Trainer API Reference

`ane_trainer` is a Python package designed to facilitate the training of small neural networks specifically targeting Apple Neural Engine (ANE) hardware. It provides a command-line interface to manage the entire training lifecycle, from data loading to model execution orchestration.

---

## Package Structure Overview

| Module | Purpose |
| :--- | :--- |
| `ane_trainer/__init__.py` | Package initialization and version exposure. |
| `ane_trainer/__version__.py` | Stores the current version of the package. |
| `ane_trainer/__main__.py` | Entry point for running the package as a script (`python -m ane_trainer`). |
| `ane_trainer/cli.py` | Defines the command-line interface using `argparse` or `click`. |
| `ane_trainer/core.py` | Contains the main training orchestration logic. |
| `ane_trainer/data.py` | Handles dataset loading, preprocessing, and batching. |
| `ane_trainer/models.py` | Defines the neural network architectures. |
| `ane_trainer/utils.py` | Utility functions (logging, device handling, etc.). |
| `example.py` | A standalone script demonstrating typical usage. |
| `tests/__init__.py` | Contains unit and integration tests. |

---

## Module Details

### `ane_trainer/__version__.py`

This module simply holds the package version string.

**Key Contents:**
*   `__version__`: `str`

**Description:**
Provides access to the installed version of the `ane_trainer` library.

**Example Usage:**
```python
from ane_trainer import __version__
print(f"ANE Trainer Version: {__version__}")
```

### `ane_trainer/__init__.py`

This module exposes core components of the package for easy importing.

**Key Contents:**
*   Imports from `core`, `data`, and `models` to allow direct access (e.g., `from ane_trainer import Trainer`).

**Description:**
Initializes the package and defines the public API surface.

**Example Usage:**
```python
from ane_trainer import Trainer, load_mnist_dataset
# Trainer and load_mnist_dataset are now directly available
```

### `ane_trainer/__main__.py`

This module allows the package to be executed directly from the command line using `python -m ane_trainer`.

**Key Contents:**
*   Calls the main execution logic defined in `cli.py`.

**Description:**
Serves as the default entry point when the package is run as a module.

**Example Usage:**
```bash
python -m ane_trainer train --dataset mnist
```

### `ane_trainer/cli.py`

This module is responsible for parsing command-line arguments and dispatching tasks to the core logic.

**Key Functions:**
*   `main(args: argparse.Namespace)`: Parses arguments and initiates the training process.

**Signature:**
```python
def main(args: argparse.Namespace) -> None:
    """
    Parses command-line arguments and starts the ANE training workflow.
    """
```

**Description:**
Sets up the CLI structure, handling flags for dataset selection, epochs, and model configuration.

**Example Usage:**
```bash
# Training MNIST with 10 epochs
ane_trainer train --dataset mnist --epochs 10
```

### `ane_trainer/core.py`

The central orchestrator of the training process. It manages the flow between data loading, model definition, and the training loop execution.

**Key Classes/Functions:**
*   `Trainer`: Manages the entire training lifecycle.

**Signature:**
```python
class Trainer:
    def __init__(self, model, data_loader, config: dict):
        """
        Initializes the Trainer with necessary components.
        """
    def train(self, epochs: int) -> dict:
        """
        Executes the training loop, orchestrating forward/backward passes.
        """
```

**Description:**
The `Trainer` class encapsulates the training loop. It interacts with `data.py` for batches and `models.py` for computation, abstracting the hardware-specific calls (ANE integration) within its execution methods.

**Example Usage:**
```python
# Assuming model and data_loader are initialized
trainer = Trainer(my_model, my_data, config)
results = trainer.train(epochs=50)
print(results)
```

### `ane_trainer/data.py`

Handles all aspects of data management, including fetching, transformation, and batching.

**Key Functions:**
*   `load_dataset(dataset_name: str) -> Dataset`: Fetches the specified dataset (e.g., MNIST).
*   `create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader`: Wraps the dataset into an iterable data loader.

**Signature:**
```python
def load_dataset(dataset_name: str) -> Dataset:
    """
    Loads and preprocesses the specified dataset (e.g., MNIST).
    """
def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """
    Creates a PyTorch/TF-compatible data loader from the dataset.
    """
```

**Description:**
Ensures data is correctly formatted and batched for efficient model consumption.

**Example Usage:**
```python
from ane_trainer.data import load_dataset, create_dataloader

mnist_data = load_dataset("mnist")
data_loader = create_dataloader(mnist_data, batch_size=32)
```

### `ane_trainer/models.py`

Defines the neural network architectures that will be trained.

**Key Classes:**
*   `SimpleNN`: A basic 2-3 layer feedforward network.

**Signature:**
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the network layers.
        """
    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass computation graph.
        """
```

**Description:**
Contains the model definitions. These models are designed to be compatible with the underlying framework used for ANE compilation/execution.

**Example Usage:**
```python
import torch.nn as nn
from ane_trainer.models import SimpleNN

# Example for MNIST (784 inputs, 10 outputs)
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
```

### `ane_trainer/utils.py`

A collection of general-purpose helper functions.

**Key Functions:**
*   `log_message(level: str, message: str)`: Standardized logging utility.
*   `check_ane_availability() -> bool`: Checks for necessary ANE runtime dependencies.

**Signature:**
```python
def log_message(level: str, message: str) -> None:
    """
    Prints a standardized, timestamped log message.
    """
def check_ane_availability() -> bool:
    """
    Verifies if the required ANE runtime environment is present.
    """
```

**Description:**
Provides cross-cutting concerns like logging and environment checks, keeping `core.py` clean.

**Example Usage:**
```python
from ane_trainer.utils import log_message

log_message("INFO", "Starting model compilation phase.")
```

---

## External Example Script

### `example.py`

A demonstration script showing how a user would typically interact with the package components directly, bypassing the CLI.

**Description:**
Illustrates the end-to-end workflow: load data $\rightarrow$ build model $\rightarrow$ train.

**Example Usage:**
```python
# example.py
from ane_trainer.data import load_dataset, create_dataloader
from ane_trainer.models import SimpleNN
from ane_trainer.core import Trainer
from ane_trainer.utils import log_message

# 1. Setup
log_message("INFO", "Initializing training pipeline...")
dataset = load_dataset("mnist")
data_loader = create_dataloader(dataset, batch_size=64)

# 2. Model Definition
model = SimpleNN(input_size=784, hidden_size=256, output_size=10)

# 3. Training Orchestration
config = {"optimizer": "Adam", "learning_rate": 0.001}
trainer = Trainer(model, data_loader, config)

# 4. Execution
log_message("INFO", "Starting training on ANE target...")
training_results = trainer.train(epochs=5)

print("\n--- Training Complete ---")
print(f"Final Loss: {training_results['final_loss']}")
```