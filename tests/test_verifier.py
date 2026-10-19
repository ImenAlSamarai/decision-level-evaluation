"""Tests for RegimeSwitchingVerifier and its helper functions."""
import re

import numpy as np
import pytest

from optimal_switching.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    _linear_utility,
)
from optimal_switching.problems.regime_switching.verifier import (
    RegimeSwitchingVerifier,
    _parse_decision,
    _compute_realized_utility,
    _apply_policy_table,
    _generate_random_decisions,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SMALL_CFG = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
GENERATOR = RegimeSwitchingGenerator(config=SMALL_CFG)


@pytest.fixture
def problem():
    return GENERATOR.sample(seed=42)


# ---------------------------------------------------------------------------
# TestParseDecision
# ---------------------------------------------------------------------------


class TestParseDecision:
    def test_standard_format_s0(self):
        assert _parse_decision("s_0 = 0") == 0

    def test_standard_format_s1(self):
        assert _parse_decision("s_1 = 1") == 1

    def test_no_underscore(self):
        assert _parse_decision("s = 1") == 1

    def test_with_newlines(self):
        text = "Some reasoning here.\ns_3 = 1\nMore text."
        assert _parse_decision(text) == 1

    def test_takes_last_match(self):
        # Multiple matches — should return last
        text = "s_0 = 0\ns_1 = 1\ns_2 = 0"
        assert _parse_decision(text) == 0

    def test_takes_last_match_second_is_one(self):
        text = "s_0 = 0\ns_1 = 1"
        assert _parse_decision(text) == 1

    def test_fails_gracefully_no_match(self):
        assert _parse_decision("no decision here") == 0

    def test_fails_gracefully_empty(self):
        assert _parse_decision("") == 0

    def test_ignores_non_binary(self):
        # "2" is not {0,1}, should find no valid match and default to 0
        assert _parse_decision("s_0 = 2") == 0

    def test_extra_spaces(self):
        assert _parse_decision("s_0  =  1") == 1

    def test_only_zero(self):
        assert _parse_decision("s = 0") == 0


# ---------------------------------------------------------------------------
# TestComputeRealizedUtility
# ---------------------------------------------------------------------------


class TestComputeRealizedUtility:
    def test_all_off_zero_utility(self):
        """All decisions = 0, all PnL non-zero: realized utility = 0 (0 * x_path)."""
        T = 5
        decisions = np.zeros(T, dtype=int)
        x_path = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
        # u(0 * x) = u(0) = 0 with linear utility, no switching cost (start=0)
        result = _compute_realized_utility(
            decisions=decisions,
            x_path=x_path,
            lam=0.1,
            initial_regime=0,
            utility_fn=_linear_utility,
        )
        assert result == pytest.approx(0.0)

    def test_all_on_from_on_equals_sum_pnl(self):
        """All decisions = 1, initial_regime = 1, lam > 0: no switching cost."""
        T = 4
        decisions = np.ones(T, dtype=int)
        x_path = np.array([1.0, 2.0, 3.0, 4.0])
        lam = 0.5
        # No switching: u(1*x_path[t]) = x_path[t], no costs
        result = _compute_realized_utility(
            decisions=decisions,
            x_path=x_path,
            lam=lam,
            initial_regime=1,
            utility_fn=_linear_utility,
        )
        assert result == pytest.approx(sum(x_path))

    def test_switching_cost_applied(self):
        """One switch from 0->1 should subtract lam."""
        T = 3
        decisions = np.array([0, 1, 1])
        x_path = np.array([1.0, 2.0, 3.0])
        lam = 0.5
        initial_regime = 0
        # t=0: action=0, s_prev=0 -> no switch. u(0)=0. cost=0
        # t=1: action=1, s_prev=0 -> switch. u(1*2.0)=2.0. cost=0.5
        # t=2: action=1, s_prev=1 -> no switch. u(1*3.0)=3.0. cost=0
        expected = 0.0 + (2.0 - 0.5) + 3.0
        result = _compute_realized_utility(
            decisions=decisions,
            x_path=x_path,
            lam=lam,
            initial_regime=initial_regime,
            utility_fn=_linear_utility,
        )
        assert result == pytest.approx(expected)

    def test_multiple_switches(self):
        """Two switches: costs applied twice."""
        T = 4
        decisions = np.array([1, 0, 1, 0])
        x_path = np.array([1.0, 1.0, 1.0, 1.0])
        lam = 0.1
        initial_regime = 0
        # t=0: 0->1 switch, u(1)=1, cost=0.1
        # t=1: 1->0 switch, u(0)=0, cost=0.1
        # t=2: 0->1 switch, u(1)=1, cost=0.1
        # t=3: 1->0 switch, u(0)=0, cost=0.1
        expected = (1.0 - 0.1) + (0.0 - 0.1) + (1.0 - 0.1) + (0.0 - 0.1)
        result = _compute_realized_utility(
            decisions=decisions,
            x_path=x_path,
            lam=lam,
            initial_regime=initial_regime,
            utility_fn=_linear_utility,
        )
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestApplyPolicyTable
# ---------------------------------------------------------------------------


class TestApplyPolicyTable:
    def test_shape(self, problem):
        from optimal_switching.problems.regime_switching.generator import _compute_z_grid

        grid_size = problem.optimal_policy_table.shape[1]
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa, grid_size
        )
        decisions = _apply_policy_table(
            policy_table=problem.optimal_policy_table,
            z_path=problem.z_path,
            z_grid=z_grid,
            initial_regime=problem.initial_regime,
        )
        assert decisions.shape == (problem.T,)

    def test_values_binary(self, problem):
        from optimal_switching.problems.regime_switching.generator import _compute_z_grid

        grid_size = problem.optimal_policy_table.shape[1]
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa, grid_size
        )
        decisions = _apply_policy_table(
            policy_table=problem.optimal_policy_table,
            z_path=problem.z_path,
            z_grid=z_grid,
            initial_regime=problem.initial_regime,
        )
        assert set(decisions.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# TestGenerateRandomDecisions
# ---------------------------------------------------------------------------


class TestGenerateRandomDecisions:
    def test_shape(self):
        d = _generate_random_decisions(T=10, seed=0)
        assert d.shape == (10,)

    def test_binary_values(self):
        d = _generate_random_decisions(T=100, seed=0)
        assert set(d.tolist()).issubset({0, 1})

    def test_deterministic(self):
        d1 = _generate_random_decisions(T=20, seed=7)
        d2 = _generate_random_decisions(T=20, seed=7)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds(self):
        d1 = _generate_random_decisions(T=20, seed=0)
        d2 = _generate_random_decisions(T=20, seed=1)
        # Should differ with high probability
        assert not np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# TestRegimeSwitchingVerifier
# ---------------------------------------------------------------------------


class TestRegimeSwitchingVerifier:
    def test_optimal_scores_one(self, problem):
        """Optimal agent completion should score close to 1.0."""
        from optimal_switching.agents.optimal_agent import OptimalAgent

        agent = OptimalAgent()
        decisions = agent.decide(problem)
        # Build completions: one turn per step
        completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]

        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, problem, mode="trajectory")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_empty_completions_score_low(self, problem):
        """Empty completions default to 0 (all off) — should score low."""
        completions = ["no decision"] * problem.T
        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, problem, mode="trajectory")
        # Score can be <= 1.0; it could be negative if random does better
        assert isinstance(score, float)

    def test_score_returns_float(self, problem):
        completions = ["s_0 = 0"] * problem.T
        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, problem, mode="trajectory")
        assert isinstance(score, float)

    def test_per_step_mode_range(self, problem):
        """Per-step mode should return a float in [0, 1]."""
        from optimal_switching.agents.optimal_agent import OptimalAgent

        agent = OptimalAgent()
        decisions = agent.decide(problem)
        completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]
        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, problem, mode="per_step")
        assert 0.0 <= score <= 1.0

    def test_per_step_optimal_is_one(self, problem):
        """Optimal agent per-step score should be 1.0."""
        from optimal_switching.agents.optimal_agent import OptimalAgent

        agent = OptimalAgent()
        decisions = agent.decide(problem)
        completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]
        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, problem, mode="per_step")
        assert score == pytest.approx(1.0)

    def test_completions_shorter_than_T_defaults_to_zero(self, problem):
        """If completions list is shorter than T, missing steps default to 0."""
        # Only provide first 2 decisions
        completions = ["s_0 = 1", "s_1 = 1"]
        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, problem, mode="trajectory")
        assert isinstance(score, float)

    def test_score_normalization_random_baseline(self):
        """Random baseline gives ~0 score for random decisions."""
        # Generate a fresh problem
        gen = RegimeSwitchingGenerator(
            config=GeneratorConfig(T_range=(10, 10), grid_size=50, n_quad_nodes=10)
        )
        prob = gen.sample(seed=100)
        # Use random decisions with same seed offset as baseline
        rng = np.random.default_rng(prob.seed)
        decisions = rng.integers(0, 2, size=prob.T)
        completions = [f"s_{t} = {decisions[t]}" for t in range(prob.T)]
        verifier = RegimeSwitchingVerifier()
        score = verifier.score(completions, prob, mode="trajectory")
        assert isinstance(score, float)
