"""Verifier for RegimeSwitchingProblem."""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from optimal_switching.base.verifier import BaseVerifier
from optimal_switching.base.problem import BaseProblem
from optimal_switching.problems.regime_switching.generator import (
    RegimeSwitchingProblem,
    _compute_z_grid,
    _linear_utility,
)


# ---------------------------------------------------------------------------
# Helper: parse a single LLM turn
# ---------------------------------------------------------------------------

def _parse_decision(text: str) -> int:
    """Extract the last binary decision from an LLM response.

    Looks for patterns like ``s_0 = 1``, ``s_t = 0``, ``s = 1``,
    ``s_17 = 0``, ``s_{-1} = 1``.
    Returns the *last* match converted to int, or 0 on failure.
    """
    # Match s with underscore + identifier (s_0, s_t, s_{-1}, s_17)
    # or standalone s = 0/1
    # The (?<![a-zA-Z]) lookbehind prevents matching "step", "state", etc.
    matches = re.findall(r"(?<![a-zA-Z])s(?:_\{?[\w\-]*\}?)?\s*=\s*([01])\b", text)
    if not matches:
        return 0
    return int(matches[-1])


# ---------------------------------------------------------------------------
# Helper: realized utility of a decision sequence
# ---------------------------------------------------------------------------

def _compute_realized_utility(
    decisions: np.ndarray,
    x_path: np.ndarray,
    lam: float,
    initial_regime: int,
    utility_fn: Callable,
) -> float:
    """Compute J(s) = Σ_t u(s[t] * x_path[t]) - lam * 1{s[t] != s[t-1]}.

    s[-1] is ``initial_regime``.
    """
    total = 0.0
    s_prev = initial_regime
    T = len(decisions)
    for t in range(T):
        a = int(decisions[t])
        reward = utility_fn(a * x_path[t])
        cost = lam if a != s_prev else 0.0
        total += reward - cost
        s_prev = a
    return float(total)


# ---------------------------------------------------------------------------
# Helper: walk the policy table along z_path
# ---------------------------------------------------------------------------

def _apply_policy_table(
    policy_table: np.ndarray,
    z_path: np.ndarray,
    z_grid: np.ndarray,
    initial_regime: int,
) -> np.ndarray:
    """Roll the policy table forward along a realised z_path.

    Args:
        policy_table: shape (T, grid_size, 2)
        z_path: shape (T+1,) — signal path (index 0..T-1 used for decisions)
        z_grid: shape (grid_size,)
        initial_regime: s_{-1}

    Returns:
        decisions: shape (T,) int array
    """
    T = policy_table.shape[0]
    decisions = np.empty(T, dtype=int)
    s_prev = initial_regime
    for t in range(T):
        z_idx = int(np.argmin(np.abs(z_grid - z_path[t])))
        a = int(policy_table[t, z_idx, s_prev])
        decisions[t] = a
        s_prev = a
    return decisions


# ---------------------------------------------------------------------------
# Helper: generate random baseline decisions
# ---------------------------------------------------------------------------

def _generate_random_decisions(T: int, seed: int) -> np.ndarray:
    """Generate seeded uniform {0,1} decisions."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=T)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class RegimeSwitchingVerifier(BaseVerifier):
    """Parses LLM completions and scores them against the optimal policy.

    Scoring (trajectory mode):
        score = (J(s_model) - J(s_random)) / max(J(s_star) - J(s_random), 1e-6)

    Per-step mode (diagnostic):
        fraction of steps where s_model[t] == optimal action given the model's
        actual s_{t-1}.
    """

    def score(
        self,
        completion: list[str],
        problem: BaseProblem,
        mode: str = "trajectory",
    ) -> float:
        assert isinstance(problem, RegimeSwitchingProblem)

        T = problem.T

        # 1. Parse decisions from completion turns
        s_model = np.empty(T, dtype=int)
        for t in range(T):
            if t < len(completion):
                s_model[t] = _parse_decision(completion[t])
            else:
                s_model[t] = 0  # default off for missing turns

        # 2. Reconstruct z_grid
        grid_size = problem.optimal_policy_table.shape[1]
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa, grid_size
        )

        # Always use linear utility for scoring (same as generator default)
        utility_fn = _linear_utility

        if mode == "per_step":
            return self._per_step_score(
                s_model, problem.optimal_policy_table, problem.z_path,
                z_grid, problem.initial_regime, T,
            )

        # 3. Realized utility for the model
        j_model = _compute_realized_utility(
            s_model, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )

        # 4. Realized utility for the optimal policy
        s_star = _apply_policy_table(
            problem.optimal_policy_table, problem.z_path,
            z_grid, problem.initial_regime,
        )
        j_star = _compute_realized_utility(
            s_star, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )

        # 5. Realized utility for random baseline
        # Average over multiple seeds to avoid single-seed collisions,
        # especially at short horizons where T=3 has only 8 possible sequences.
        n_random_seeds = 20
        j_randoms = []
        for k in range(n_random_seeds):
            s_rand_k = _generate_random_decisions(T, problem.seed * 1000 + k)
            j_rand_k = _compute_realized_utility(
                s_rand_k, problem.x_path, problem.lam,
                problem.initial_regime, utility_fn,
            )
            j_randoms.append(j_rand_k)
        j_random = float(np.mean(j_randoms))

        # 6. Regret-normalised score
        gap = j_star - j_random
        if gap <= 1e-6:
            # Degenerate instance — optimal and random are indistinguishable.
            # If model matches optimal exactly, score 1.0 (preserves invariant).
            # Otherwise score 0.0 (uninformative instance).
            if abs(j_model - j_star) < 1e-10:
                return 1.0
            return 0.0
        raw_score = (j_model - j_random) / gap
        # Clip to prevent outliers from tiny denominators dominating means.
        # Does not affect GRPO (which normalises within groups).
        return float(np.clip(raw_score, -2.0, 2.0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _per_step_score(
        self,
        s_model: np.ndarray,
        policy_table: np.ndarray,
        z_path: np.ndarray,
        z_grid: np.ndarray,
        initial_regime: int,
        T: int,
    ) -> float:
        """Fraction of steps where the model matches the optimal action,
        given the model's own trajectory (actual s_{t-1})."""
        correct = 0
        s_prev = initial_regime
        for t in range(T):
            z_idx = int(np.argmin(np.abs(z_grid - z_path[t])))
            optimal_a = int(policy_table[t, z_idx, s_prev])
            if int(s_model[t]) == optimal_a:
                correct += 1
            s_prev = int(s_model[t])
        return float(correct) / T
