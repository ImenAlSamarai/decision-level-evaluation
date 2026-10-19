"""Greedy one-step lookahead validation agent."""
from __future__ import annotations

import numpy as np

from optimal_switching.base.problem import BaseProblem
from optimal_switching.problems.regime_switching.generator import RegimeSwitchingProblem


class GreedyAgent:
    """At each step t, picks the action maximising immediate Q-value.

    Q(a, s_prev) = alpha * z_path[t] * a  -  lam * 1{a != s_prev}
    """

    def decide(self, problem: BaseProblem) -> np.ndarray:
        """Return shape (T,) integer array of greedy decisions."""
        assert isinstance(problem, RegimeSwitchingProblem)
        T = problem.T
        decisions = np.empty(T, dtype=int)
        s_prev = problem.initial_regime
        for t in range(T):
            z = problem.z_path[t]
            q_off = 0.0 - (problem.lam if 0 != s_prev else 0.0)
            q_on = problem.alpha * z - (problem.lam if 1 != s_prev else 0.0)
            a = 1 if q_on > q_off else 0
            decisions[t] = a
            s_prev = a
        return decisions
