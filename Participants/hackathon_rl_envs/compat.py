"""Optional Gymnasium compatibility helpers."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces

    BaseEnv = gym.Env
    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - default path in the repo test env
    class BaseEnv:
        """Small subset of the Gymnasium env interface."""

        metadata = {}

        def reset(self, *, seed=None, options=None):
            self._last_seed = seed
            return None, {}

    class Discrete:
        def __init__(self, n):
            self.n = int(n)

        def sample(self):
            return int(np.random.randint(0, self.n))

    class Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class _Spaces:
        Discrete = Discrete
        Box = Box

    spaces = _Spaces()
    GYMNASIUM_AVAILABLE = False
