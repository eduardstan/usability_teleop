"""Shared permutation-test configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PermutationConfig:
    """Permutation test settings."""

    n_permutations: int = 200
    alpha: float = 0.05
    random_seed: int = 42
    nested: bool = False
