"""Random validation agent — uniform {0,1} decisions."""
from __future__ import annotations

import numpy as np

from optimal_switching.base.problem import BaseProblem
from optimal_switching.problems.regime_switching.generator import RegimeSwitchingProblem


class RandomAgent:
    """Samples uniform {0,1} decisions seeded by ``problem.seed + seed_offset``."""

    def __init__(self, seed_offset: int = 1) -> None:
        self.seed_offset = seed_offset

    def decide(self, problem: BaseProblem) -> np.ndarray:
        """Return shape (T,) int8 array of random binary decisions."""
        assert isinstance(problem, RegimeSwitchingProblem)
        rng = np.random.default_rng(problem.seed + self.seed_offset)
        return rng.integers(0, 2, size=problem.T)
