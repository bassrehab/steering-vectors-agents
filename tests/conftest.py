"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Get the best available device for testing."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"
