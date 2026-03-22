"""Shared utilities for the local RL environments and server games."""

from __future__ import annotations

import hashlib
import random


def clamp(value, low, high):
    """Clamp a scalar to a closed interval."""
    return max(low, min(high, value))


def system_seed():
    """Return a fresh seed suitable for per-episode hidden randomization."""
    return random.SystemRandom().randrange(1, 2**31)


def make_step_rng(seed, step, namespace):
    """Create a deterministic RNG for a given episode seed and step."""
    payload = f"{int(seed)}:{int(step)}:{namespace}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return random.Random(int.from_bytes(digest, "big"))
