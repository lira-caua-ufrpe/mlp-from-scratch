"""Utility functions."""


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)
