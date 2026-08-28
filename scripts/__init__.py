"""Command-line entry points with deterministic CUDA process configuration."""

import os

# Package initialization runs before entry-point modules import PyTorch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
