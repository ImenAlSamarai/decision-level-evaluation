"""Tests for RegimeSwitchingProblem, GeneratorConfig, Bellman solver, and RegimeSwitchingGenerator."""
import numpy as np
import pytest

from optimal_switching.problems.regime_switching.generator import (
    _linear_utility,
    _exponential_utility,
    _compute_z_grid,
    _solve_bellman,
    RegimeSwitchingProblem,
    GeneratorConfig,
    RegimeSwitchingGenerator,
)


# ---------------------------------------------------------------------------
# TestUtilityFunctions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_linear_utility_zero(self):
        assert _linear_utility(0.0) == 0.0

    def test_linear_utility_positive(self):
        assert _linear_utility(3.5) == 3.5

    def test_linear_utility_negative(self):
        assert _linear_utility(-2.0) == -2.0

    def test_exponential_utility_at_zero(self):
        # u(0) = 1 - exp(0) = 0
        assert _exponential_utility(0.0, gamma=1.0) == pytest.approx(0.0)

    def test_exponential_utility_positive(self):
        # u(1) = 1 - exp(-1)
        expected = 1.0 - np.exp(-1.0)
        assert _exponential_utility(1.0, gamma=1.0) == pytest.approx(expected)

    def test_exponential_utility_large_x_approaches_one(self):
        val = _exponential_utility(100.0, gamma=1.0)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_exponential_utility_custom_gamma(self):
        gamma = 2.0
        x = 0.5
        expected = 1.0 - np.exp(-gamma * x)
        assert _exponential_utility(x, gamma=gamma) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestZGrid
# ---------------------------------------------------------------------------

class TestZGrid:
    def test_shape(self):
        z = _compute_z_grid(theta=0.0, sigma_z=0.2, kappa=0.3, grid_size=50)
        assert z.shape == (50,)

    def test_centered_on_theta(self):
        theta = 0.5
        z = _compute_z_grid(theta=theta, sigma_z=0.2, kappa=0.3, grid_size=51)
        # midpoint should equal theta
        assert z[25] == pytest.approx(theta)

    def test_spans_four_std(self):
        theta = 0.0
        sigma_z = 0.2
        kappa = 0.3
        stationary_std = sigma_z / np.sqrt(2 * kappa)
        z = _compute_z_grid(theta=theta, sigma_z=sigma_z, kappa=kappa, grid_size=100)
        assert z[0] == pytest.approx(theta - 4 * stationary_std)
        assert z[-1] == pytest.approx(theta + 4 * stationary_std)

    def test_monotone_increasing(self):
        z = _compute_z_grid(theta=0.0, sigma_z=0.2, kappa=0.3, grid_size=50)
        assert np.all(np.diff(z) > 0)


# ---------------------------------------------------------------------------
# TestRegimeSwitchingProblem
# ---------------------------------------------------------------------------

class TestRegimeSwitchingProblem:
    def _make_problem(self, T=5, grid_size=50):
        z_path = np.zeros(T + 1)
        x_path = np.zeros(T)
        policy = np.zeros((T, grid_size, 2), dtype=int)
        value = 0.5
        return RegimeSwitchingProblem(
            kappa=0.3,
            theta=0.0,
            sigma_z=0.2,
            alpha=0.3,
            sigma_x=0.2,
            lam=0.1,
            T=T,
            seed=42,
            z_path=z_path,
            x_path=x_path,
            initial_regime=0,
            optimal_policy_table=policy,
            optimal_value=value,
        )

    def test_horizon(self):
        prob = self._make_problem(T=5)
        assert prob.horizon() == 5

    def test_z_path_shape(self):
        T = 5
        prob = self._make_problem(T=T)
        assert prob.z_path.shape == (T + 1,)

    def test_x_path_shape(self):
        T = 5
        prob = self._make_problem(T=T)
        assert prob.x_path.shape == (T,)

    def test_policy_table_shape(self):
        T = 5
        grid_size = 50
        prob = self._make_problem(T=T, grid_size=grid_size)
        assert prob.optimal_policy_table.shape == (T, grid_size, 2)

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(RegimeSwitchingProblem)

    def test_fields_accessible(self):
        prob = self._make_problem()
        assert prob.kappa == 0.3
        assert prob.theta == 0.0
        assert prob.lam == 0.1
        assert prob.initial_regime == 0
        assert prob.optimal_value == 0.5


# ---------------------------------------------------------------------------
# TestGeneratorConfig
# ---------------------------------------------------------------------------

class TestGeneratorConfig:
    def test_default_config(self):
        cfg = GeneratorConfig()
        assert cfg.theta == 0.0
        assert cfg.utility == "linear"
        assert cfg.gamma == 1.0
        assert cfg.grid_size == 200
        assert cfg.n_quad_nodes == 20

    def test_alpha_min_enforced(self):
        with pytest.raises(ValueError, match="alpha_min must be >= 0.1"):
            GeneratorConfig(alpha_range=(0.05, 0.5))

    def test_alpha_min_exactly_0_1_is_valid(self):
        cfg = GeneratorConfig(alpha_range=(0.1, 0.5))
        assert cfg.alpha_range[0] == 0.1

    def test_custom_config(self):
        cfg = GeneratorConfig(
            kappa_range=(0.2, 0.4),
            theta=1.0,
            sigma_z_range=(0.15, 0.25),
            utility="exponential",
            gamma=2.0,
            grid_size=100,
        )
        assert cfg.theta == 1.0
        assert cfg.utility == "exponential"
        assert cfg.gamma == 2.0
        assert cfg.grid_size == 100

    def test_default_ranges(self):
        cfg = GeneratorConfig()
        assert cfg.kappa_range == (0.1, 0.5)
        assert cfg.T_range == (3, 20)
        assert cfg.lam_range == (0.0, 0.3)


# ---------------------------------------------------------------------------
# TestBellmanSolver
# ---------------------------------------------------------------------------

SMALL_GRID = 50
SMALL_QUAD = 10


class TestBellmanSolver:
    def _z_grid(self, T=4, grid_size=SMALL_GRID):
        return _compute_z_grid(theta=0.0, sigma_z=0.2, kappa=0.3, grid_size=grid_size)

    def _linear_fn(self, x):
        return _linear_utility(x)

    def test_policy_shape(self):
        z_grid = self._z_grid()
        T = 4
        policy, value = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=SMALL_QUAD,
        )
        assert policy.shape == (T, SMALL_GRID, 2)

    def test_value_shape(self):
        z_grid = self._z_grid()
        T = 4
        policy, value = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=SMALL_QUAD,
        )
        assert value.shape == (T + 1, SMALL_GRID, 2)

    def test_no_switching_cost_high_z_action_is_1(self):
        """With lam=0 and high z, action should be 1 (buy signal)."""
        z_grid = self._z_grid()
        T = 3
        policy, value = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=SMALL_QUAD,
        )
        # At t=0, z_max, both s_prev=0 and s_prev=1 should choose action=1
        assert policy[0, -1, 0] == 1
        assert policy[0, -1, 1] == 1

    def test_no_switching_cost_low_z_action_is_0(self):
        """With lam=0 and low z, action should be 0."""
        z_grid = self._z_grid()
        T = 3
        policy, value = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=SMALL_QUAD,
        )
        # At t=0, z_min, both s_prev=0 and s_prev=1 should choose action=0
        assert policy[0, 0, 0] == 0
        assert policy[0, 0, 1] == 0

    def test_very_high_switching_cost_forces_stay(self):
        """With very high lam, at z_mid, agent stays with current regime."""
        z_grid = self._z_grid()
        T = 3
        lam = 10.0
        policy, value = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=lam,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=SMALL_QUAD,
        )
        mid_idx = SMALL_GRID // 2
        # s_prev=0 should stay at 0
        assert policy[0, mid_idx, 0] == 0
        # s_prev=1 should stay at 1
        assert policy[0, mid_idx, 1] == 1

    def test_terminal_value_is_zero(self):
        """Terminal value table at t=T should be zeros."""
        z_grid = self._z_grid()
        T = 3
        policy, value = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=SMALL_QUAD,
        )
        np.testing.assert_array_equal(value[T], np.zeros((SMALL_GRID, 2)))


# ---------------------------------------------------------------------------
# TestRegimeSwitchingGenerator
# ---------------------------------------------------------------------------

class TestRegimeSwitchingGenerator:
    def _make_generator(self):
        cfg = GeneratorConfig(
            T_range=(4, 6),
            grid_size=SMALL_GRID,
            n_quad_nodes=SMALL_QUAD,
        )
        return RegimeSwitchingGenerator(config=cfg)

    def test_sample_returns_problem(self):
        gen = self._make_generator()
        prob = gen.sample(seed=0)
        assert isinstance(prob, RegimeSwitchingProblem)

    def test_sample_z_path_shape(self):
        gen = self._make_generator()
        prob = gen.sample(seed=0)
        assert prob.z_path.shape == (prob.T + 1,)

    def test_sample_x_path_shape(self):
        gen = self._make_generator()
        prob = gen.sample(seed=0)
        assert prob.x_path.shape == (prob.T,)

    def test_sample_policy_shape(self):
        gen = self._make_generator()
        prob = gen.sample(seed=0)
        assert prob.optimal_policy_table.shape == (prob.T, SMALL_GRID, 2)

    def test_same_seed_same_problem(self):
        gen = self._make_generator()
        prob1 = gen.sample(seed=42)
        prob2 = gen.sample(seed=42)
        assert prob1.T == prob2.T
        np.testing.assert_array_equal(prob1.z_path, prob2.z_path)
        np.testing.assert_array_equal(prob1.x_path, prob2.x_path)
        assert prob1.optimal_value == prob2.optimal_value

    def test_different_seed_different_problem(self):
        gen = self._make_generator()
        prob1 = gen.sample(seed=0)
        prob2 = gen.sample(seed=1)
        # Very likely to differ in some parameter
        different = (
            prob1.T != prob2.T
            or not np.allclose(prob1.z_path, prob2.z_path)
            or not np.allclose(prob1.x_path, prob2.x_path)
        )
        assert different

    def test_default_config_used_when_none(self):
        gen = RegimeSwitchingGenerator()
        assert gen.config is not None
        assert isinstance(gen.config, GeneratorConfig)

    def test_horizon_matches_T(self):
        gen = self._make_generator()
        prob = gen.sample(seed=7)
        assert prob.horizon() == prob.T

    def test_initial_regime_is_valid(self):
        gen = self._make_generator()
        prob = gen.sample(seed=3)
        assert prob.initial_regime in (0, 1)

    def test_seed_stored_in_problem(self):
        gen = self._make_generator()
        prob = gen.sample(seed=99)
        assert prob.seed == 99

    def test_optimal_value_is_float(self):
        gen = self._make_generator()
        prob = gen.sample(seed=5)
        assert isinstance(prob.optimal_value, float)

    def test_exponential_utility_config(self):
        cfg = GeneratorConfig(
            T_range=(3, 4),
            grid_size=SMALL_GRID,
            n_quad_nodes=SMALL_QUAD,
            utility="exponential",
            gamma=1.5,
        )
        gen = RegimeSwitchingGenerator(config=cfg)
        prob = gen.sample(seed=10)
        assert isinstance(prob, RegimeSwitchingProblem)
