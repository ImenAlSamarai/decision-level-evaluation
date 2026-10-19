"""Regime-switching problem generator with Bellman solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.special import roots_hermite

from optimal_switching.base.generator import BaseGenerator
from optimal_switching.base.problem import BaseProblem


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _linear_utility(x: float) -> float:
    """Risk-neutral utility: u(x) = x."""
    return x


def _exponential_utility(x: float, gamma: float) -> float:
    """Risk-averse utility: u(x) = 1 - exp(-gamma * x)."""
    return 1.0 - np.exp(-gamma * x)


# ---------------------------------------------------------------------------
# Z-grid computation
# ---------------------------------------------------------------------------

def _compute_z_grid(theta: float, sigma_z: float, kappa: float, grid_size: int) -> np.ndarray:
    """Compute problem-specific z-grid spanning ±4 std of OU stationary distribution.

    Stationary distribution is N(theta, sigma_z^2 / (2*kappa)).
    """
    stationary_std = sigma_z / np.sqrt(2 * kappa)
    z_min = theta - 4 * stationary_std
    z_max = theta + 4 * stationary_std
    return np.linspace(z_min, z_max, grid_size)


# ---------------------------------------------------------------------------
# RegimeSwitchingProblem dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeSwitchingProblem(BaseProblem):
    kappa: float          # OU mean-reversion speed
    theta: float          # OU long-run mean
    sigma_z: float        # OU signal volatility (discrete-time, per step)
    alpha: float          # Signal strength: E[X_{t+1} | Z_t] = alpha * Z_t
    sigma_x: float        # PnL noise std dev
    lam: float            # Switching cost lambda
    T: int                # Horizon
    seed: int             # Random seed
    z_path: np.ndarray    # shape (T+1,) — observable signal
    x_path: np.ndarray    # shape (T,)   — realised PnL increments
    initial_regime: int   # s_{-1}, regime before episode starts
    optimal_policy_table: np.ndarray   # shape (T, grid_size, 2)
    optimal_value: float               # E[J(s*)]

    def horizon(self) -> int:
        return self.T


# ---------------------------------------------------------------------------
# GeneratorConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Configuration for the regime switching generator.

    Parameters are sampled uniformly from their ranges. Two modes for
    setting the switching cost:
      - lam_range: sample lambda independently (default)
      - lam_alpha_ratio_range: sample lambda/alpha ratio, then compute
        lambda = alpha * ratio. Ensures the ratio stays in a controlled
        range. If set, lam_range is ignored.
    """
    kappa_range: tuple[float, float] = (0.1, 0.5)
    theta: float = 0.0
    sigma_z_range: tuple[float, float] = (0.1, 0.3)
    alpha_range: tuple[float, float] = (0.1, 0.5)
    sigma_x_range: tuple[float, float] = (0.1, 0.3)
    lam_range: tuple[float, float] = (0.0, 0.3)
    lam_alpha_ratio_range: tuple[float, float] | None = None
    T_range: tuple[int, int] = (3, 20)
    utility: Literal["linear", "exponential"] = "linear"
    gamma: float = 1.0
    grid_size: int = 200
    n_quad_nodes: int = 20

    def __post_init__(self) -> None:
        if self.alpha_range[0] < 0.1:
            raise ValueError(
                f"alpha_min must be >= 0.1 to prevent degenerate instances, "
                f"got alpha_range[0]={self.alpha_range[0]}"
            )

    @classmethod
    def planning_zone(cls) -> "GeneratorConfig":
        """Config targeting the planning zone identified by the parameter sweep.

        kappa in [0.1, 0.25] (persistent signals where planning matters),
        lambda/alpha ratio in [0.3, 0.5] (intermediate switching costs),
        other parameters vary freely from defaults.
        """
        return cls(
            kappa_range=(0.1, 0.25),
            alpha_range=(0.1, 0.5),
            lam_alpha_ratio_range=(0.3, 0.5),
            sigma_z_range=(0.1, 0.3),
            sigma_x_range=(0.1, 0.3),
            T_range=(5, 20),
        )

    @classmethod
    def j_gap_zone(cls) -> "GeneratorConfig":
        """Config targeting large J gap (J_optimal - J_greedy) for GRPO training.

        Found via parameter sweep over 3,024 configs x 20 episodes.
        In this zone greedy captures only 30-60% of J_optimal, giving
        GRPO a strong economic signal to learn planning.

        Key drivers of J gap:
          - Low kappa (persistent signals -> planning over many steps)
          - High lambda/alpha ratio (costly switches -> greedy over-trades)
          - Long horizon (more steps for planning advantage to compound)
        """
        return cls(
            kappa_range=(0.01, 0.10),
            alpha_range=(0.1, 0.5),
            lam_alpha_ratio_range=(2.0, 5.0),
            sigma_z_range=(0.1, 0.5),
            sigma_x_range=(0.1, 0.3),
            T_range=(50, 100),
        )

    @classmethod
    def control_zone(cls) -> "GeneratorConfig":
        """Config for the control zone where greedy ≈ optimal.

        kappa = 0.7 (fast reversion, greedy is near-optimal).
        Same other parameter ranges as planning_zone for fair comparison.
        """
        return cls(
            kappa_range=(0.7, 0.7),
            alpha_range=(0.1, 0.5),
            lam_alpha_ratio_range=(0.3, 0.5),
            sigma_z_range=(0.1, 0.3),
            sigma_x_range=(0.1, 0.3),
            T_range=(5, 20),
        )


# ---------------------------------------------------------------------------
# Bellman solver
# ---------------------------------------------------------------------------

def _solve_bellman(
    z_grid: np.ndarray,
    T: int,
    alpha: float,
    lam: float,
    kappa: float,
    theta: float,
    sigma_z: float,
    utility_fn: Callable,
    n_quad_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Bellman recursion via backward induction.

    Returns:
        policy_table: shape (T, grid_size, 2) — optimal action at (t, z_idx, s_prev)
        value_table: shape (T+1, grid_size, 2) — V_t(z, s_prev)
    """
    grid_size = len(z_grid)

    # Gauss-Hermite nodes and weights
    gh_nodes, gh_weights = roots_hermite(n_quad_nodes)

    # Initialise tables
    value_table = np.zeros((T + 1, grid_size, 2))
    policy_table = np.zeros((T, grid_size, 2), dtype=int)

    # Backward induction: t = T-1 down to 0
    for t in range(T - 1, -1, -1):
        v_next = value_table[t + 1]  # shape (grid_size, 2)

        for z_idx in range(grid_size):
            z = z_grid[z_idx]

            # OU transition: E[z'] = mu, with noise sigma_z
            mu = z + kappa * (theta - z)

            # GH quadrature points: z' = mu + sqrt(2)*sigma_z*node
            z_prime_points = mu + np.sqrt(2.0) * sigma_z * gh_nodes

            for s_prev in range(2):
                best_q = -np.inf
                best_a = 0

                for a in range(2):
                    # Immediate reward: u(a * alpha * z)
                    reward = utility_fn(a * alpha * z)

                    # Switching cost
                    cost = lam if a != s_prev else 0.0

                    # Expected continuation value via GH quadrature
                    v_interp = np.interp(z_prime_points, z_grid, v_next[:, a])
                    continuation = (1.0 / np.sqrt(np.pi)) * np.dot(gh_weights, v_interp)

                    q_val = reward - cost + continuation

                    if q_val > best_q:
                        best_q = q_val
                        best_a = a

                policy_table[t, z_idx, s_prev] = best_a
                value_table[t, z_idx, s_prev] = best_q

    return policy_table, value_table


# ---------------------------------------------------------------------------
# RegimeSwitchingGenerator
# ---------------------------------------------------------------------------

class RegimeSwitchingGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()

    def sample(self, seed: int) -> RegimeSwitchingProblem:
        """Generate a single fully-solved RegimeSwitchingProblem instance."""
        cfg = self.config
        rng = np.random.default_rng(seed)

        # 1. Sample DGP parameters from config ranges
        kappa = float(rng.uniform(*cfg.kappa_range))
        theta = cfg.theta
        sigma_z = float(rng.uniform(*cfg.sigma_z_range))
        alpha = float(rng.uniform(*cfg.alpha_range))
        sigma_x = float(rng.uniform(*cfg.sigma_x_range))
        if cfg.lam_alpha_ratio_range is not None:
            ratio = float(rng.uniform(*cfg.lam_alpha_ratio_range))
            lam = alpha * ratio
        else:
            lam = float(rng.uniform(*cfg.lam_range))
        T = int(rng.integers(cfg.T_range[0], cfg.T_range[1] + 1))
        initial_regime = int(rng.integers(0, 2))

        # 2. Simulate OU path: z[t+1] = z[t] + kappa*(theta - z[t]) + sigma_z*eps
        #    z[0] drawn from stationary distribution N(theta, sigma_z^2/(2*kappa))
        stationary_std = sigma_z / np.sqrt(2 * kappa)
        z_path = np.empty(T + 1)
        z_path[0] = float(rng.normal(theta, stationary_std))
        for t in range(T):
            z_path[t + 1] = z_path[t] + kappa * (theta - z_path[t]) + sigma_z * rng.standard_normal()

        # 3. Simulate PnL: x[t] = alpha * z[t] + sigma_x * eps
        x_path = alpha * z_path[:T] + sigma_x * rng.standard_normal(T)

        # 4. Compute z_grid and solve Bellman
        z_grid = _compute_z_grid(theta=theta, sigma_z=sigma_z, kappa=kappa, grid_size=cfg.grid_size)

        # Build utility function (possibly partial application for exponential)
        if cfg.utility == "linear":
            utility_fn = _linear_utility
        else:
            gamma = cfg.gamma

            def utility_fn(x: float) -> float:
                return _exponential_utility(x, gamma)

        policy_table, value_table = _solve_bellman(
            z_grid=z_grid,
            T=T,
            alpha=alpha,
            lam=lam,
            kappa=kappa,
            theta=theta,
            sigma_z=sigma_z,
            utility_fn=utility_fn,
            n_quad_nodes=cfg.n_quad_nodes,
        )

        # 5. Compute optimal value: E[J(s*)] by rolling forward the optimal policy
        #    using z_path and starting from initial_regime
        total_reward = 0.0
        s_prev = initial_regime
        for t in range(T):
            z_idx = int(np.argmin(np.abs(z_grid - z_path[t])))
            a = int(policy_table[t, z_idx, s_prev])
            reward = utility_fn(a * alpha * z_path[t])
            cost = lam if a != s_prev else 0.0
            total_reward += reward - cost
            s_prev = a

        optimal_value = float(total_reward)

        return RegimeSwitchingProblem(
            kappa=kappa,
            theta=theta,
            sigma_z=sigma_z,
            alpha=alpha,
            sigma_x=sigma_x,
            lam=lam,
            T=T,
            seed=seed,
            z_path=z_path,
            x_path=x_path,
            initial_regime=initial_regime,
            optimal_policy_table=policy_table,
            optimal_value=optimal_value,
        )
