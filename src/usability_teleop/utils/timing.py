"""Timing and progress utilities for long-running pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    total = int(max(0, round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class ProgressTracker:
    """Track elapsed time and ETA for iterative tasks."""

    total: int
    completed: int = 0

    def __post_init__(self) -> None:
        self._start = time.perf_counter()

    def step(self) -> tuple[float, float]:
        """Increment progress and return (elapsed_sec, eta_sec)."""
        self.completed += 1
        elapsed = time.perf_counter() - self._start
        if self.completed <= 0:
            return elapsed, float("inf")
        rate = elapsed / self.completed
        remaining = max(self.total - self.completed, 0)
        eta = rate * remaining
        return elapsed, eta

