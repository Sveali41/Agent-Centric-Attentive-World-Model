"""Uniform transition-level reservoir replay for streaming MiniGrid data."""

from __future__ import annotations

import os
import tempfile
from typing import Dict

import numpy as np
import torch

from .fisher_buffer import FisherReplayBuffer


class ReservoirReplayBuffer(FisherReplayBuffer):
    """Keep a uniform sample of all transitions seen in a stream.

    The parent class is retained for the shared aligned-array I/O helpers used
    by the training pipeline.  MiniGrid never calls its domain-salience
    insertion path: every transition is offered to this reservoir exactly
    once, and all transition fields are replaced atomically.
    """

    def __init__(self, max_size: int, seed: int = 0):
        super().__init__(max_size=max_size, contact_positive_ratio=0.0)
        if int(max_size) <= 0:
            raise ValueError(f"Reservoir capacity must be positive, got {max_size!r}")
        self.max_size = int(max_size)
        self.num_seen = 0
        self.rng = np.random.default_rng(int(seed))

    def update_combined(
        self,
        samples: Dict,
        current_sample_ratio: float = 1.0,
        fisher_buffer_elements_ratio: float = 0.0,
        target_shape=None,
    ) -> None:
        """Offer every input transition to the reservoir.

        The legacy ratio arguments remain accepted so existing trainer call
        sites can migrate without changing the generic buffer interface, but
        they have no effect on MiniGrid reservoir admission.
        """
        del current_sample_ratio, fisher_buffer_elements_ratio, target_shape
        if samples is None or "obs" not in samples:
            raise KeyError("Reservoir samples must contain 'obs'")

        total_len = len(samples["obs"])
        for index in range(total_len):
            sample = self._sample_at(samples, index)
            self.num_seen += 1

            if len(self.buffer) < self.max_size:
                self.buffer.append(sample)
                continue

            replacement = int(self.rng.integers(0, self.num_seen))
            if replacement < self.max_size:
                self.buffer[replacement] = sample

    def save_to_file(self, path: str) -> None:
        """Persist samples and reservoir state for an exact resume."""
        target = os.path.abspath(os.path.expanduser(str(path)))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = {
            "data": self.export_dict(),
            "num_seen": int(self.num_seen),
            "rng_state": self.rng.bit_generator.state,
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.",
            suffix=".tmp",
            dir=os.path.dirname(target),
        )
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_from_file(self, path: str) -> None:
        """Load an exact reservoir checkpoint created by ``save_to_file``."""
        target = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.exists(target):
            raise FileNotFoundError(f"No reservoir file found at {target}")
        payload = torch.load(target, weights_only=False)
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError("Reservoir checkpoint is missing its state payload")
        self.load_from_dict(payload["data"])
        self.num_seen = max(int(payload.get("num_seen", len(self.buffer))), len(self.buffer))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
