"""Unit tests for ane_trainer utilities and core functionality."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ane_trainer.data import load_dataset
from ane_trainer.models import build_model, SimpleNN
from ane_trainer.utils import is_apple_silicon


class TestIsAppleSilicon(unittest.TestCase):
    """Tests for is_apple_silicon() platform detection function."""

    def test_is_apple_silicon_returns_bool(self):
        """Verify that is_apple_silicon() returns a bool type."""
        result = is_apple_silicon()
        self.assertIsInstance(result, bool)

    def test_is_apple_silicon_non_apple_returns_false(self):
        """Verify False is returned on non-Apple hardware (Linux, Windows, x86)."""
        # Mock platform.system() to return 'Linux' (non-Apple)
        with patch("platform.system", return_value="Linux"):
            with patch("platform.processor", return_value="x86_64"):
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_mocked_true_arm64(self):
        """Test is_apple_silicon returns True when mocking Apple Silicon with arm64."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="arm64"):
                result = is_apple_silicon()
                self.assertTrue(result)

    def test_is_apple_silicon_mocked_true_arm(self):
        """Test is_apple_silicon returns True when mocking Apple Silicon with arm."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="arm"):
                result = is_apple_silicon()
                self.assertTrue(result)

    def test_is_apple_silicon_mocked_false_darwin_x86(self):
        """Test is_apple_silicon returns False for Darwin with x86 processor."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="i386"):
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_mocked_false_windows(self):
        """Test is_apple_silicon returns False for Windows platform."""
        with patch("platform.system", return_value="Windows"):
            with patch("platform.processor", return_value="AMD64"):
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_case_insensitive_arm64(self):
        """Test is_apple_silicon is case-insensitive for arm64 detection."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="ARM64"):
                result = is_apple_silicon()
                self.assertTrue(result)

    def test_is_apple_silicon_case_insensitive_arm(self):
        """Test is_apple_silicon is case-insensitive for arm detection."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="ARM"):
                result = is_apple_silicon()
                self.assertTrue(result)

    def test_is_apple_silicon_with_whitespace(self):
        """Test is_apple_silicon handles processor string with whitespace."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value=" arm64 "):
                result = is_apple_silicon()
                self.assertTrue(result)

    def test_is_apple_silicon_empty_processor(self):
        """Test is_apple_silicon handles empty processor string gracefully."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value=""):
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_none_processor(self):
        """Test is_apple_silicon handles None processor gracefully."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value=None):
                # None will raise AttributeError on .lower(), but should be caught
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_exception_handling_platform_system(self):
        """Test is_apple_silicon handles exceptions from platform.system()."""
        with patch("platform.system", side_effect=Exception("Platform error")):
            with patch("platform.processor", return_value="arm64"):
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_exception_handling_processor(self):
        """Test is_apple_silicon handles exceptions from platform.processor()."""
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", side_effect=Exception("Processor error")):
                result = is_apple_silicon()
                self.assertFalse(result)

    def test_is_apple_silicon_deterministic(self):
        """Test that is_apple_silicon returns the same value on repeated calls."""
        # Call multiple times and verify consistent results
        result1 = is_apple_silicon()
        result2 = is_apple_silicon()
        result3 = is_apple_silicon()
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)

    def test_is_apple_silicon_no_exceptions_raised(self):
        """Verify that is_apple_silicon() raises no exceptions on any platform."""
        # Even with unusual inputs, no exception should be raised
        with patch("platform.system", return_value="UnknownOS"):
            with patch("platform.processor", return_value="UnknownProcessor"):
                try:
                    result = is_apple_silicon()
                    # If we get here without exception, test passes
                    self.assertIsInstance(result, bool)
                except Exception as e:
                    self.fail(f"is_apple_silicon() raised {type(e).__name__}: {e}")

    def test_is_apple_silicon_combined_darwin_and_arm64(self):
        """Test both conditions must be true: Darwin AND arm64."""
        # Darwin but not arm64
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="x86_64"):
                self.assertFalse(is_apple_silicon())

        # arm64 but not Darwin
        with patch("platform.system", return_value="Linux"):
            with patch("platform.processor", return_value="arm64"):
                self.assertFalse(is_apple_silicon())

        # Both Darwin and arm64
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.processor", return_value="arm64"):
                self.assertTrue(is_apple_silicon())


class TestLoadDataset:
    """Tests for load_dataset() function."""

    def test_load_dataset_shapes(self):
        """Verify X_train and y_train have correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X_train, y_train, X_test, y_test = load_dataset(tmpdir)

            assert X_train.shape == (60000, 28, 28), f"X_train shape: {X_train.shape}"
            assert y_train.shape == (60000,), f"y_train shape: {y_train.shape}"
            assert X_test.shape == (10000, 28, 28), f"X_test shape: {X_test.shape}"
            assert y_test.shape == (10000,), f"y_test shape: {y_test.shape}"

    def test_load_dataset_dtype_range(self):
        """Verify data types and value ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X_train, y_train, X_test, y_test = load_dataset(tmpdir)

            # Check dtypes
            assert X_train.dtype == np.float32, f"X_train dtype: {X_train.dtype}"
            assert X_test.dtype == np.float32, f"X_test dtype: {X_test.dtype}"
            assert y_train.dtype == np.int64, f"y_train dtype: {y_train.dtype}"
            assert y_test.dtype == np.int64, f"y_test dtype: {y_test.dtype}"

            # Check value ranges
            assert X_train.min() >= 0.0 and X_train.max() <= 1.0, \
                f"X_train range: [{X_train.min()}, {X_train.max()}]"
            assert X_test.min() >= 0.0 and X_test.max() <= 1.0, \
                f"X_test range: [{X_test.min()}, {X_test.max()}]"

            # Check label ranges
            assert set(np.unique(y_train)).issubset(set(range(10))), \
                f"y_train labels: {np.unique(y_train)}"
            assert set(np.unique(y_test)).issubset(set(range(10))), \
                f"y_test labels: {np.unique(y_test)}"

    def test_load_dataset_creates_directory(self):
        """Verify load_dataset creates dataset_path if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "mnist_data")
            assert not os.path.exists(dataset_path)

            X_train, y_train, X_test, y_test = load_dataset(dataset_path)

            assert os.path.exists(dataset_path), f"Directory not created: {dataset_path}"

    def test_load_dataset_caches_files(self):
        """Verify load_dataset uses cache on subsequent calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First call downloads data
            X_train1, y_train1, X_test1, y_test1 = load_dataset(tmpdir)

            # Count files in the MNIST directory
            mnist_dir = os.path.join(tmpdir, "MNIST")
            if os.path.exists(mnist_dir):
                files_after_first = sum(
                    len(files)
                    for _, _, files in os.walk(mnist_dir)
                )
            else:
                files_after_first = 0

            # Second call should use cache (no re-download)
            X_train2, y_train2, X_test2, y_test2 = load_dataset(tmpdir)

            # Verify data is identical (cache reused)
            np.testing.assert_array_equal(X_train1, X_train2)
            np.testing.assert_array_equal(y_train1, y_train2)
            np.testing.assert_array_equal(X_test1, X_test2)
            np.testing.assert_array_equal(y_test1, y_test2)

            # Count files again
            if os.path.exists(mnist_dir):
                files_after_second = sum(
                    len(files)
                    for _, _, files in os.walk(mnist_dir)
                )
            else:
                files_after_second = 0

            # Should be same number of files (no re-download)
            assert files_after_first == files_after_second, \
                "Files changed between calls (re-download occurred)"

    def test_load_dataset_invalid_parent_directory(self):
        """Verify load_dataset raises OSError for invalid parent directory."""
        invalid_path = "/nonexistent/parent/mnist_data"
        with pytest.raises(OSError, match="Dataset path parent must exist"):
            load_dataset(invalid_path)

    def test_load_dataset_download_failure_no_cache(self):
        """Verify load_dataset raises RuntimeError if download fails and no cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock torchvision to fail
            with mock.patch(
                "torchvision.datasets.MNIST",
                side_effect=Exception("Network error")
            ):
                with pytest.raises(RuntimeError, match="Failed to download MNIST"):
                    load_dataset(tmpdir)

    def test_load_dataset_return_tuple(self):
        """Verify load_dataset returns 4-tuple."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_dataset(tmpdir)
            assert isinstance(result, tuple), f"Result type: {type(result)}"
            assert len(result) == 4, f"Result length: {len(result)}"


class TestBuildModel:
    """Tests for build_model() function."""

    def test_build_model_valid(self, toy_model):
        """Verify build_model returns torch.nn.Module with forward method."""
        assert isinstance(toy_model, torch.nn.Module), \
            f"Model should be torch.nn.Module, got {type(toy_model)}"
        assert hasattr(toy_model, 'forward'), \
            "Model must have forward() method"
        assert callable(toy_model.forward), \
            "forward must be callable"

    def test_build_model_returns_simple_nn(self):
        """Verify build_model returns SimpleNN instance."""
        model = build_model(784, 128, 10)
        assert isinstance(model, SimpleNN), \
            f"build_model should return SimpleNN instance, got {type(model)}"

    def test_build_model_output_shape_batch_32(self):
        """Verify output shape (32, 784) input → (32, 10) output."""
        model = build_model(784, 128, 10)
        model.eval()

        x = torch.randn(32, 784)
        output = model(x)

        assert output.shape == (32, 10), \
            f"Expected output shape (32, 10), got {output.shape}"
        assert output.dtype == torch.float32, \
            f"Expected dtype float32, got {output.dtype}"

    def test_build_model_output_shape_batch_1(self):
        """Verify output shape with batch size 1."""
        model = build_model(784, 128, 10)
        model.eval()

        x = torch.randn(1, 784)
        output = model(x)

        assert output.shape == (1, 10), \
            f"Expected output shape (1, 10), got {output.shape}"
        assert output.dtype == torch.float32, \
            f"Expected dtype float32, got {output.dtype}"

    def test_build_model_output_shape_batch_64(self):
        """Verify output shape with batch size 64."""
        model = build_model(784, 128, 10)
        model.eval()

        x = torch.randn(64, 784)
        output = model(x)

        assert output.shape == (64, 10), \
            f"Expected output shape (64, 10), got {output.shape}"
        assert output.dtype == torch.float32, \
            f"Expected dtype float32, got {output.dtype}"

    def test_build_model_trainable(self, toy_model):
        """Verify all model parameters have requires_grad=True."""
        for param in toy_model.parameters():
            assert param.requires_grad is True, \
                f"All parameters must be trainable (requires_grad=True), found requires_grad={param.requires_grad}"

    def test_build_model_parameter_count(self):
        """Verify model has expected number of parameters."""
        model = build_model(784, 128, 10)

        # Expected parameters:
        # fc1: 784 * 128 weights + 128 biases = 100480
        # fc2: 128 * 10 weights + 10 biases = 1290
        # Total: 101770
        total_params = sum(p.numel() for p in model.parameters())
        expected_params = (784 * 128 + 128) + (128 * 10 + 10)

        assert total_params == expected_params, \
            f"Expected {expected_params} parameters, got {total_params}"

    def test_build_model_deterministic(self):
        """Verify model produces same output for same input in eval mode."""
        model = build_model(784, 128, 10)
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_build_model_relu_activation(self):
        """Verify ReLU activation is applied in hidden layer."""
        model = build_model(10, 5, 2)
        model.eval()

        x = torch.randn(2, 10)
        # Hook to capture ReLU output
        relu_output = None

        def hook_relu(module, input, output):
            nonlocal relu_output
            relu_output = output

        model.relu.register_forward_hook(hook_relu)

        with torch.no_grad():
            _ = model(x)

        # Verify ReLU output is non-negative (ReLU property)
        assert (relu_output >= 0).all(), "ReLU output should be non-negative"


class TestAneForwardPass:
    """Tests for ane_forward_pass() function."""

    def test_ane_forward_numpy_io(self, toy_model, toy_data):
        """Test ane_forward_pass accepts numpy input and returns numpy output with correct shape."""
        from ane_trainer.core import ane_forward_pass

        X, _ = toy_data
        toy_model.eval()

        output = ane_forward_pass(toy_model, X)

        # Verify output is numpy array
        assert isinstance(output, np.ndarray), f"Output type: {type(output)}"
        # Verify output shape
        assert output.shape == (64, 10), f"Output shape: {output.shape}"
        # Verify output dtype
        assert output.dtype == np.float32, f"Output dtype: {output.dtype}"

    def test_ane_forward_non_apple(self, toy_model):
        """Test ane_forward_pass works on non-Apple hardware (mock is_apple_silicon to False)."""
        from ane_trainer.core import ane_forward_pass

        X = np.random.randn(8, 784).astype(np.float32)
        X = np.clip(X, 0, 1)
        toy_model.eval()

        # Mock is_apple_silicon to return False
        with patch('ane_trainer.core.is_apple_silicon', return_value=False):
            output = ane_forward_pass(toy_model, X)

        # Should not raise exception and produce valid output
        assert isinstance(output, np.ndarray)
        assert output.shape == (8, 10)
        assert output.dtype == np.float32

    def test_ane_forward_deterministic(self, toy_model):
        """Test ane_forward_pass is deterministic: same input → same output."""
        from ane_trainer.core import ane_forward_pass

        X = np.random.randn(4, 784).astype(np.float32)
        X = np.clip(X, 0, 1)
        toy_model.eval()

        out1 = ane_forward_pass(toy_model, X)
        out2 = ane_forward_pass(toy_model, X)

        # Same input should produce identical output
        np.testing.assert_allclose(out1, out2)

    def test_ane_forward_invalid_shape(self, toy_model):
        """Test ane_forward_pass raises ValueError for invalid input shape."""
        from ane_trainer.core import ane_forward_pass

        # Wrong number of features (not 784)
        X_wrong = np.random.randn(32, 100).astype(np.float32)
        toy_model.eval()

        with pytest.raises(ValueError, match="Expected input shape"):
            ane_forward_pass(toy_model, X_wrong)

    def test_ane_forward_invalid_dtype(self, toy_model):
        """Test ane_forward_pass raises ValueError for invalid input dtype."""
        from ane_trainer.core import ane_forward_pass

        # Wrong dtype (int32 instead of float32)
        X_wrong = np.random.randint(0, 100, (32, 784)).astype(np.int32)
        toy_model.eval()

        with pytest.raises(ValueError, match="Expected input dtype"):
            ane_forward_pass(toy_model, X_wrong)

    def test_ane_forward_sets_eval_mode(self, toy_model):
        """Test that ane_forward_pass sets model to eval mode."""
        from ane_trainer.core import ane_forward_pass

        X = np.random.randn(8, 784).astype(np.float32)
        X = np.clip(X, 0, 1)

        # Set model to training mode
        toy_model.train()
        assert toy_model.training is True

        # Forward pass should set to eval mode
        ane_forward_pass(toy_model, X)

        # After forward pass, model should be in eval mode
        assert toy_model.training is False


class TestTrainStep:
    """Tests for train_step() function."""

    def test_train_step_returns_scalar(self, toy_model, toy_data, optimizer, loss_fn):
        """Test train_step returns a float scalar loss value > 0."""
        from ane_trainer.core import train_step

        X, y = toy_data

        loss = train_step(toy_model, X, y, optimizer, loss_fn)

        # Verify type is float
        assert isinstance(loss, float), f"Loss type: {type(loss)}"
        # Verify loss is positive
        assert loss > 0, f"Loss should be positive: {loss}"
        # Verify loss is finite
        assert np.isfinite(loss), f"Loss should be finite: {loss}"

    def test_train_step_updates_weights(self, toy_model, toy_data, optimizer, loss_fn):
        """Test train_step updates model weights."""
        from ane_trainer.core import train_step

        X, y = toy_data

        # Capture initial weights
        initial_weights = [p.clone().detach() for p in toy_model.parameters()]

        # Perform training step
        train_step(toy_model, X, y, optimizer, loss_fn)

        # Verify weights have changed
        for init_w, curr_w in zip(initial_weights, toy_model.parameters()):
            assert not torch.allclose(init_w, curr_w), \
                "Weights should be updated after train_step"

    def test_train_step_decreases_loss(self, toy_model, toy_data, optimizer, loss_fn):
        """Test train_step decreases loss over iterations on constant ground truth."""
        from ane_trainer.core import train_step

        X, _ = toy_data
        # Fixed target (all class 0)
        y = np.zeros(X.shape[0], dtype=np.int64)

        # Perform multiple training steps
        loss1 = train_step(toy_model, X, y, optimizer, loss_fn)
        loss2 = train_step(toy_model, X, y, optimizer, loss_fn)
        loss3 = train_step(toy_model, X, y, optimizer, loss_fn)

        # Loss should generally decrease (may have some oscillation)
        assert loss1 > loss2, f"Loss should decrease: {loss1} -> {loss2}"
        assert loss2 > loss3 or loss2 >= loss3 * 0.99, \
            f"Loss should not increase significantly: {loss2} -> {loss3}"

    def test_train_step_batch_size_mismatch(self, toy_model, optimizer, loss_fn):
        """Test train_step raises ValueError for mismatched batch sizes."""
        from ane_trainer.core import train_step

        X = np.random.randn(32, 784).astype(np.float32)
        y = np.random.randint(0, 10, 64).astype(np.int64)  # Different size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            train_step(toy_model, X, y, optimizer, loss_fn)

    def test_train_step_invalid_y_dtype(self, toy_model, optimizer, loss_fn):
        """Test train_step raises ValueError for invalid y dtype."""
        from ane_trainer.core import train_step

        X = np.random.randn(32, 784).astype(np.float32)
        y = np.random.randint(0, 10, 32).astype(np.float32)  # Wrong dtype

        with pytest.raises(ValueError, match="Expected y dtype int64"):
            train_step(toy_model, X, y, optimizer, loss_fn)

    def test_train_step_invalid_y_range(self, toy_model, optimizer, loss_fn):
        """Test train_step raises ValueError for y values outside [0, 9]."""
        from ane_trainer.core import train_step

        X = np.random.randn(32, 784).astype(np.float32)
        y = np.random.randint(0, 15, 32).astype(np.int64)  # Some values > 9

        with pytest.raises(ValueError, match="Expected y values in range"):
            train_step(toy_model, X, y, optimizer, loss_fn)

    def test_train_step_negative_y_values(self, toy_model, optimizer, loss_fn):
        """Test train_step raises ValueError for negative y values."""
        from ane_trainer.core import train_step

        X = np.random.randn(32, 784).astype(np.float32)
        y = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8] * 3 + [9, 9], dtype=np.int64)  # Has -1

        with pytest.raises(ValueError, match="Expected y values in range"):
            train_step(toy_model, X, y, optimizer, loss_fn)

    def test_train_step_batch_size_one(self, toy_model, optimizer, loss_fn):
        """Test train_step works with batch size 1."""
        from ane_trainer.core import train_step

        X = np.random.randn(1, 784).astype(np.float32)
        y = np.array([5], dtype=np.int64)

        loss = train_step(toy_model, X, y, optimizer, loss_fn)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_sets_train_mode(self, toy_model, toy_data, optimizer, loss_fn):
        """Test that train_step sets model to training mode."""
        from ane_trainer.core import train_step

        X, y = toy_data

        # Set model to eval mode
        toy_model.eval()
        assert toy_model.training is False

        # train_step should set model to train mode during execution
        train_step(toy_model, X, y, optimizer, loss_fn)

        # After train_step, model is typically left in train mode
        assert toy_model.training is True


class TestMainFunction(unittest.TestCase):
    """Tests for main() CLI training orchestration function."""

    def _create_mock_dataset(self):
        """Create a mock MNIST dataset with toy data."""
        X_train = np.random.randn(128, 28, 28).astype(np.float32)  # Small dataset
        X_train = np.clip(X_train, 0, 1)
        y_train = np.random.randint(0, 10, 128).astype(np.int64)
        X_test = np.random.randn(32, 28, 28).astype(np.float32)
        X_test = np.clip(X_test, 0, 1)
        y_test = np.random.randint(0, 10, 32).astype(np.int64)
        return X_train, y_train, X_test, y_test

    def test_main_saves_model(self):
        """Test that main() saves model to file and file is non-empty."""
        from ane_trainer.cli import main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "mnist_data")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            output_path = os.path.join(output_dir, "model.pt")

            args = argparse.Namespace(
                dataset=dataset_path,
                epochs=1,
                output=output_path,
                batch_size=32,
                learning_rate=0.01
            )

            # Mock load_dataset to return toy data
            mock_dataset = self._create_mock_dataset()
            with patch("ane_trainer.cli.load_dataset", return_value=mock_dataset):
                exit_code = main(args)

            assert exit_code == 0, f"main() returned {exit_code}, expected 0"
            assert os.path.isfile(output_path), f"Model file not found at {output_path}"
            assert os.path.getsize(output_path) > 0, "Model file is empty"

    def test_main_logs_loss_format(self):
        """Test that main() logs loss in correct format: 'Epoch N: Loss F.FFFF'."""
        from ane_trainer.cli import main
        import argparse
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "mnist_data")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            output_path = os.path.join(output_dir, "model.pt")

            args = argparse.Namespace(
                dataset=dataset_path,
                epochs=2,
                output=output_path,
                batch_size=32,
                learning_rate=0.01
            )

            mock_dataset = self._create_mock_dataset()
            stdout_capture = io.StringIO()
            with patch("ane_trainer.cli.load_dataset", return_value=mock_dataset):
                with redirect_stdout(stdout_capture):
                    exit_code = main(args)

            output_text = stdout_capture.getvalue()

            # Verify we got exactly 2 epoch logs
            import re
            epoch_lines = re.findall(r"Epoch \d+: Loss \d+\.\d{4}", output_text)
            assert len(epoch_lines) == 2, f"Expected 2 epoch logs, got {len(epoch_lines)}: {output_text}"

            # Verify format of each line
            lines = output_text.strip().split('\n')
            for i, line in enumerate(lines):
                expected_pattern = f"Epoch {i + 1}: Loss"
                assert line.startswith(expected_pattern), f"Line {i} doesn't match expected format: {line}"

    def test_main_exits_zero_on_success(self):
        """Test that main() returns 0 on successful training."""
        from ane_trainer.cli import main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "mnist_data")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            output_path = os.path.join(output_dir, "model.pt")

            args = argparse.Namespace(
                dataset=dataset_path,
                epochs=1,
                output=output_path,
                batch_size=32,
                learning_rate=0.01
            )

            mock_dataset = self._create_mock_dataset()
            with patch("ane_trainer.cli.load_dataset", return_value=mock_dataset):
                exit_code = main(args)

            assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

    def test_main_exits_one_on_missing_dataset(self):
        """Test that main() returns 1 when dataset path is nonexistent."""
        from ane_trainer.cli import main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a nonexistent dataset path
            dataset_path = os.path.join(tmpdir, "nonexistent", "mnist_data")

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            output_path = os.path.join(output_dir, "model.pt")

            args = argparse.Namespace(
                dataset=dataset_path,
                epochs=1,
                output=output_path,
                batch_size=32,
                learning_rate=0.01
            )

            # Capture stderr to avoid printing error messages during test
            import io
            from contextlib import redirect_stderr

            stderr_capture = io.StringIO()
            with redirect_stderr(stderr_capture):
                exit_code = main(args)

            assert exit_code == 1, f"Expected exit code 1, got {exit_code}"

    def test_main_exits_one_on_missing_output_dir(self):
        """Test that main() returns 1 when output directory does not exist."""
        from ane_trainer.cli import main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "mnist_data")

            # Use an output path with a nonexistent parent directory
            output_path = os.path.join(tmpdir, "nonexistent", "model.pt")

            args = argparse.Namespace(
                dataset=dataset_path,
                epochs=1,
                output=output_path,
                batch_size=32,
                learning_rate=0.01
            )

            # Capture stderr
            import io
            from contextlib import redirect_stderr

            stderr_capture = io.StringIO()
            with redirect_stderr(stderr_capture):
                exit_code = main(args)

            assert exit_code == 1, f"Expected exit code 1, got {exit_code}"


class TestMainCli(unittest.TestCase):
    """Tests for main_cli() argument parser function."""

    def test_cli_help_text(self):
        """Test that --help exits 0 and contains required argument names."""
        from ane_trainer.cli import main_cli
        import subprocess
        import os

        # Run with --help using subprocess to capture exit code
        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--help"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Exit code should be 0
        assert result.returncode == 0, f"--help exited with code {result.returncode}"

        # Help text should contain all required arguments
        help_text = result.stdout
        assert "--dataset" in help_text, "Help should mention --dataset"
        assert "--epochs" in help_text, "Help should mention --epochs"
        assert "--output" in help_text, "Help should mention --output"
        assert "--batch-size" in help_text, "Help should mention --batch-size"
        assert "--learning-rate" in help_text, "Help should mention --learning-rate"

    def test_cli_missing_required_arg_dataset(self):
        """Test that missing --dataset causes argparse to exit 2."""
        import subprocess
        import os

        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--epochs", "1", "--output", "/tmp/model.pt"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Exit code should be 2 (argparse error)
        assert result.returncode == 2, f"Missing --dataset should exit with code 2, got {result.returncode}"

        # Error message should be in stderr
        assert "--dataset" in result.stderr or "required" in result.stderr, \
            f"Error message should mention --dataset or 'required'. stderr: {result.stderr}"

    def test_cli_missing_required_arg_epochs(self):
        """Test that missing --epochs causes argparse to exit 2."""
        import subprocess
        import os

        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--dataset", "/tmp/mnist", "--output", "/tmp/model.pt"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Exit code should be 2 (argparse error)
        assert result.returncode == 2, f"Missing --epochs should exit with code 2, got {result.returncode}"

    def test_cli_missing_required_arg_output(self):
        """Test that missing --output causes argparse to exit 2."""
        import subprocess
        import os

        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Exit code should be 2 (argparse error)
        assert result.returncode == 2, f"Missing --output should exit with code 2, got {result.returncode}"

    def test_cli_argument_types_epochs_int(self):
        """Test that --epochs parses as int and fails with non-int."""
        import subprocess
        import os

        # Valid int
        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "abc", "--output", "/tmp/model.pt"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Non-int should cause argparse error (exit 2)
        assert result.returncode == 2, f"Non-int --epochs should exit 2, got {result.returncode}"

    def test_cli_argument_types_batch_size_int(self):
        """Test that --batch-size parses as int."""
        import subprocess
        import os

        # Non-int should cause argparse error
        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1", "--output", "/tmp/model.pt", "--batch-size", "xyz"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        assert result.returncode == 2, f"Non-int --batch-size should exit 2, got {result.returncode}"

    def test_cli_argument_types_learning_rate_float(self):
        """Test that --learning-rate parses as float."""
        import subprocess
        import os

        # Non-float should cause argparse error
        result = subprocess.run(
            [sys.executable, "-m", "ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1", "--output", "/tmp/model.pt", "--learning-rate", "not_a_float"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        assert result.returncode == 2, f"Non-float --learning-rate should exit 2, got {result.returncode}"

    def test_cli_optional_defaults_batch_size(self):
        """Test that --batch-size defaults to 32."""
        from ane_trainer.cli import main_cli
        from unittest.mock import patch
        import argparse

        # Mock sys.argv and main() to capture the args
        captured_args = None

        def mock_main(args):
            nonlocal captured_args
            captured_args = args
            return 0

        with patch("sys.argv", ["ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1", "--output", "/tmp/model.pt"]):
            with patch("ane_trainer.cli.main", side_effect=mock_main):
                main_cli()

        assert captured_args is not None, "Args not captured"
        assert captured_args.batch_size == 32, f"Expected batch_size=32, got {captured_args.batch_size}"

    def test_cli_optional_defaults_learning_rate(self):
        """Test that --learning-rate defaults to 0.01."""
        from ane_trainer.cli import main_cli
        from unittest.mock import patch
        import argparse

        captured_args = None

        def mock_main(args):
            nonlocal captured_args
            captured_args = args
            return 0

        with patch("sys.argv", ["ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1", "--output", "/tmp/model.pt"]):
            with patch("ane_trainer.cli.main", side_effect=mock_main):
                main_cli()

        assert captured_args is not None, "Args not captured"
        assert captured_args.learning_rate == 0.01, f"Expected learning_rate=0.01, got {captured_args.learning_rate}"

    def test_cli_full_invocation_with_mocked_main(self):
        """Test full invocation with mocked main() to avoid expensive training."""
        from ane_trainer.cli import main_cli
        from unittest.mock import patch
        import argparse

        captured_args = None
        mock_return_code = 0

        def mock_main(args):
            nonlocal captured_args
            captured_args = args
            return mock_return_code

        with patch("sys.argv", ["ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "2", "--output", "/tmp/model.pt", "--batch-size", "16", "--learning-rate", "0.05"]):
            with patch("ane_trainer.cli.main", side_effect=mock_main):
                exit_code = main_cli()

        # Verify exit code is propagated
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        # Verify args were parsed correctly
        assert captured_args is not None, "Args not captured"
        assert captured_args.dataset == "/tmp/mnist", f"Expected dataset=/tmp/mnist, got {captured_args.dataset}"
        assert captured_args.epochs == 2, f"Expected epochs=2, got {captured_args.epochs}"
        assert captured_args.output == "/tmp/model.pt", f"Expected output=/tmp/model.pt, got {captured_args.output}"
        assert captured_args.batch_size == 16, f"Expected batch_size=16, got {captured_args.batch_size}"
        assert captured_args.learning_rate == 0.05, f"Expected learning_rate=0.05, got {captured_args.learning_rate}"

    def test_cli_propagates_main_exit_code(self):
        """Test that main_cli() propagates exit code from main()."""
        from ane_trainer.cli import main_cli
        from unittest.mock import patch

        def mock_main(args):
            return 42  # Non-zero exit code

        with patch("sys.argv", ["ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1", "--output", "/tmp/model.pt"]):
            with patch("ane_trainer.cli.main", side_effect=mock_main):
                exit_code = main_cli()

        assert exit_code == 42, f"Expected exit code 42, got {exit_code}"

    def test_cli_signature_returns_int(self):
        """Test that main_cli() returns int type."""
        from ane_trainer.cli import main_cli
        from unittest.mock import patch

        def mock_main(args):
            return 0

        with patch("sys.argv", ["ane_trainer", "--dataset", "/tmp/mnist", "--epochs", "1", "--output", "/tmp/model.pt"]):
            with patch("ane_trainer.cli.main", side_effect=mock_main):
                result = main_cli()

        assert isinstance(result, int), f"main_cli() should return int, got {type(result)}"


if __name__ == "__main__":
    unittest.main()
