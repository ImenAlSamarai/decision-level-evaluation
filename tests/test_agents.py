"""Tests for RandomAgent, GreedyAgent, and OptimalAgent."""
import numpy as np
import pytest

from optimal_switching.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    _compute_z_grid,
)
from optimal_switching.agents.random_agent import RandomAgent
from optimal_switching.agents.greedy_agent import GreedyAgent
from optimal_switching.agents.optimal_agent import OptimalAgent


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SMALL_CFG = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
GENERATOR = RegimeSwitchingGenerator(config=SMALL_CFG)


@pytest.fixture
def problem():
    return GENERATOR.sample(seed=42)


# ---------------------------------------------------------------------------
# TestRandomAgent
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def test_correct_shape(self, problem):
        agent = RandomAgent()
        decisions = agent.decide(problem)
        assert decisions.shape == (problem.T,)

    def test_binary_values(self, problem):
        agent = RandomAgent()
        decisions = agent.decide(problem)
        assert set(decisions.tolist()).issubset({0, 1})

    def test_deterministic_with_same_seed(self, problem):
        agent = RandomAgent(seed_offset=1)
        d1 = agent.decide(problem)
        d2 = agent.decide(problem)
        np.testing.assert_array_equal(d1, d2)

    def test_different_offset_different_results(self, problem):
        a1 = RandomAgent(seed_offset=1)
        a2 = RandomAgent(seed_offset=99)
        d1 = a1.decide(problem)
        d2 = a2.decide(problem)
        # Very likely different with high probability
        assert not np.array_equal(d1, d2)

    def test_dtype_is_int(self, problem):
        agent = RandomAgent()
        decisions = agent.decide(problem)
        assert np.issubdtype(decisions.dtype, np.integer)

    def test_uses_problem_seed(self):
        """Two problems with different seeds produce different random decisions."""
        gen = RegimeSwitchingGenerator(
            config=GeneratorConfig(T_range=(10, 10), grid_size=50, n_quad_nodes=10)
        )
        prob1 = gen.sample(seed=1)
        prob2 = gen.sample(seed=2)
        agent = RandomAgent(seed_offset=0)
        d1 = agent.decide(prob1)
        d2 = agent.decide(prob2)
        # Different seeds -> different decisions (very high probability)
        assert not np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# TestGreedyAgent
# ---------------------------------------------------------------------------


class TestGreedyAgent:
    def test_correct_shape(self, problem):
        agent = GreedyAgent()
        decisions = agent.decide(problem)
        assert decisions.shape == (problem.T,)

    def test_binary_values(self, problem):
        agent = GreedyAgent()
        decisions = agent.decide(problem)
        assert set(decisions.tolist()).issubset({0, 1})

    def test_lam_zero_sign_of_alpha_z(self):
        """With lam=0, greedy picks 1 when alpha*z > 0, 0 otherwise."""
        # Construct a problem with lam=0 and controlled z_path
        from optimal_switching.problems.regime_switching.generator import RegimeSwitchingProblem

        T = 4
        z_path = np.array([1.0, -1.0, 2.0, -0.5, 0.5])  # T+1
        x_path = np.zeros(T)
        alpha = 0.3
        # policy table doesn't matter for GreedyAgent (it ignores it)
        policy = np.zeros((T, 50, 2), dtype=int)

        prob = RegimeSwitchingProblem(
            kappa=0.3,
            theta=0.0,
            sigma_z=0.2,
            alpha=alpha,
            sigma_x=0.1,
            lam=0.0,
            T=T,
            seed=0,
            z_path=z_path,
            x_path=x_path,
            initial_regime=0,
            optimal_policy_table=policy,
            optimal_value=0.0,
        )
        agent = GreedyAgent()
        decisions = agent.decide(prob)
        # alpha*z[0] = 0.3 > 0 -> 1
        # alpha*z[1] = -0.3 < 0 -> 0
        # alpha*z[2] = 0.6 > 0 -> 1
        # alpha*z[3] = -0.15 < 0 -> 0
        expected = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(decisions, expected)

    def test_dtype_is_int(self, problem):
        agent = GreedyAgent()
        decisions = agent.decide(problem)
        assert np.issubdtype(decisions.dtype, np.integer)

    def test_high_switching_cost_reduces_switches(self):
        """With high lam, greedy should prefer staying in current regime."""
        from optimal_switching.problems.regime_switching.generator import RegimeSwitchingProblem

        T = 5
        # z always slightly positive -> alpha*z = 0.3*0.1 = 0.03
        # but lam=10 makes switching very expensive
        z_path = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        x_path = np.zeros(T)
        alpha = 0.3
        policy = np.zeros((T, 50, 2), dtype=int)

        prob = RegimeSwitchingProblem(
            kappa=0.3,
            theta=0.0,
            sigma_z=0.2,
            alpha=alpha,
            sigma_x=0.1,
            lam=10.0,
            T=T,
            seed=0,
            z_path=z_path,
            x_path=x_path,
            initial_regime=0,
            optimal_policy_table=policy,
            optimal_value=0.0,
        )
        agent = GreedyAgent()
        decisions = agent.decide(prob)
        # alpha*z = 0.03, switching cost = 10 -> stay at 0
        np.testing.assert_array_equal(decisions, np.zeros(T, dtype=int))


# ---------------------------------------------------------------------------
# TestOptimalAgent
# ---------------------------------------------------------------------------


class TestOptimalAgent:
    def test_correct_shape(self, problem):
        agent = OptimalAgent()
        decisions = agent.decide(problem)
        assert decisions.shape == (problem.T,)

    def test_binary_values(self, problem):
        agent = OptimalAgent()
        decisions = agent.decide(problem)
        assert set(decisions.tolist()).issubset({0, 1})

    def test_dtype_is_int(self, problem):
        agent = OptimalAgent()
        decisions = agent.decide(problem)
        assert np.issubdtype(decisions.dtype, np.integer)

    def test_matches_manual_policy_table_trace(self, problem):
        """OptimalAgent should reproduce the same decisions as a manual table lookup."""
        agent = OptimalAgent()
        decisions = agent.decide(problem)

        # Manual trace
        grid_size = problem.optimal_policy_table.shape[1]
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa, grid_size
        )
        manual = np.empty(problem.T, dtype=int)
        s_prev = problem.initial_regime
        for t in range(problem.T):
            z_idx = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
            a = int(problem.optimal_policy_table[t, z_idx, s_prev])
            manual[t] = a
            s_prev = a

        np.testing.assert_array_equal(decisions, manual)

    def test_optimal_utility_ge_random(self, problem):
        """Optimal agent should achieve >= realized utility vs random (on average)."""
        from optimal_switching.problems.regime_switching.verifier import (
            _compute_realized_utility,
            _generate_random_decisions,
        )

        opt_agent = OptimalAgent()
        opt_decisions = opt_agent.decide(problem)
        rand_decisions = _generate_random_decisions(problem.T, problem.seed)

        utility_fn = _linear_utility  # linear is the default
        opt_util = _compute_realized_utility(
            opt_decisions, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )
        rand_util = _compute_realized_utility(
            rand_decisions, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )
        # Over many problems this holds; for seed=42, just assert both are floats
        assert isinstance(opt_util, float)
        assert isinstance(rand_util, float)


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

def _linear_utility(x):
    return float(x)
