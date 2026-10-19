"""Optimal validation agent — walks the pre-solved policy table."""
from __future__ import annotations

import numpy as np

from optimal_switching.base.problem import BaseProblem
from optimal_switching.problems.regime_switching.generator import (
    RegimeSwitchingProblem,
    _compute_z_grid,
)


class OptimalAgent:
    """Reproduces the optimal policy by looking up the policy table at each step."""

    def decide(self, problem: BaseProblem) -> np.ndarray:
        """Return shape (T,) integer array of optimal decisions."""
        assert isinstance(problem, RegimeSwitchingProblem)
        T = problem.T
        grid_size = problem.optimal_policy_table.shape[1]
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa, grid_size
        )
        decisions = np.empty(T, dtype=int)
        s_prev = problem.initial_regime
        for t in range(T):
            z_idx = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
            a = int(problem.optimal_policy_table[t, z_idx, s_prev])
            decisions[t] = a
            s_prev = a
        return decisions
