import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed supported RNGs and optionally require deterministic PyTorch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by PyTorch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create an explicitly seeded generator for a DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
