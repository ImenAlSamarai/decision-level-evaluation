"""Goldilocks validation suite for the optimal-switching benchmark.

Runs three difficulty levels (easy / medium / hard) against three agents
(random / greedy / optimal) and checks that raw performance is "just right":

- Optimal always has highest mean J
- Greedy beats random but not optimal
- The fraction of optimal value captured by greedy decreases with difficulty
- Random scores near zero

Uses RAW realised utility J(s) for validation (not normalised scores).
Normalised scores support training use; raw J is used for quality checks of the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimal_switching.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    _compute_z_grid,
)
from optimal_switching.problems.regime_switching.verifier import (
    _compute_realized_utility,
    _linear_utility,
)
from optimal_switching.agents.random_agent import RandomAgent
from optimal_switching.agents.greedy_agent import GreedyAgent
from optimal_switching.agents.optimal_agent import OptimalAgent


# ---------------------------------------------------------------------------
# Difficulty levels — kappa (mean-reversion speed) is the key difficulty knob.
# Fast reversion creates frequent switching dilemmas where planning matters.
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS: dict[str, dict] = {
    "easy": {
        "T_range": (10, 10),
        "lam_range": (0.05, 0.05),
        "alpha_range": (0.30, 0.30),
        "kappa_range": (0.3, 0.3),    # slow reversion → greedy ≈ optimal
    },
    "medium": {
        "T_range": (15, 15),
        "lam_range": (0.10, 0.10),
        "alpha_range": (0.30, 0.30),
        "kappa_range": (0.5, 0.5),    # moderate reversion → planning helps
    },
    "hard": {
        "T_range": (25, 25),
        "lam_range": (0.15, 0.15),
        "alpha_range": (0.30, 0.30),
        "kappa_range": (0.7, 0.7),    # fast reversion + long horizon → planning essential
    },
}


# ---------------------------------------------------------------------------
# GoldilocksReport
# ---------------------------------------------------------------------------

@dataclass
class GoldilocksReport:
    """Aggregated results of a Goldilocks validation run.

    Uses raw mean J(s) values, not normalised scores.
    """
    difficulty_levels: list[str]
    agents: list[str]
    mean_j: dict[str, dict[str, float]]   # agent → level → mean J(s)

    def greedy_capture_pct(self) -> dict[str, float]:
        """Percentage of optimal value captured by greedy at each level."""
        result = {}
        for level in self.difficulty_levels:
            j_opt = self.mean_j["optimal"][level]
            j_gre = self.mean_j["greedy"][level]
            if abs(j_opt) < 1e-10:
                result[level] = 100.0
            else:
                result[level] = (j_gre / j_opt) * 100.0
        return result

    def all_pass(self) -> bool:
        """Return True iff all Goldilocks criteria are met.

        1. Mean J(optimal) > Mean J(greedy) at all levels.
        2. Mean J(greedy) > Mean J(random) at all levels.
        3. Greedy capture % decreases with difficulty (planning matters more).
        4. Mean J(random) < Mean J(optimal) at all levels.
        """
        mj = self.mean_j
        captures = self.greedy_capture_pct()

        # 1. Optimal > Greedy at all levels
        for level in self.difficulty_levels:
            if mj["optimal"][level] <= mj["greedy"][level]:
                return False

        # 2. Greedy > Random at all levels
        for level in self.difficulty_levels:
            if mj["greedy"][level] <= mj["random"][level]:
                return False

        # 3. Greedy capture decreases: easy > medium > hard
        cap_values = [captures[level] for level in self.difficulty_levels]
        for i in range(len(cap_values) - 1):
            if cap_values[i] <= cap_values[i + 1]:
                return False

        return True

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append("=" * 65)
        lines.append("Goldilocks Validation Report (raw mean J values)")
        lines.append("=" * 65)

        col_w = 14
        header = f"{'Agent':<10}" + "".join(
            f"{lvl:>{col_w}}" for lvl in self.difficulty_levels
        )
        lines.append(header)
        lines.append("-" * len(header))

        for agent in self.agents:
            row = f"{agent:<10}"
            for level in self.difficulty_levels:
                j = self.mean_j[agent][level]
                row += f"{j:>{col_w}.4f}"
            lines.append(row)

        lines.append("-" * len(header))

        # Greedy capture row
        captures = self.greedy_capture_pct()
        cap_row = f"{'capture%':<10}"
        for level in self.difficulty_levels:
            cap_row += f"{captures[level]:>{col_w - 1}.1f}%"
        lines.append(cap_row)

        lines.append("=" * 65)
        verdict = "PASS" if self.all_pass() else "FAIL"
        lines.append(f"Overall: {verdict}")

        cap_list = [captures[l] for l in self.difficulty_levels]
        mono = all(cap_list[i] > cap_list[i+1] for i in range(len(cap_list)-1))
        lines.append(f"Greedy capture decreasing: {'YES' if mono else 'NO'} "
                      f"({' → '.join(f'{c:.1f}%' for c in cap_list)})")
        lines.append("=" * 65)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GoldilocksValidator
# ---------------------------------------------------------------------------

class GoldilocksValidator:
    """Run the Goldilocks validation suite using raw J(s) comparisons.

    Unlike normalised scoring (used for GRPO), this uses raw realised
    utility to avoid normalisation artifacts.
    """

    def __init__(
        self,
        n_instances: int = 500,
        grid_size: int = 200,
        n_quad_nodes: int = 20,
    ) -> None:
        self.n_instances = n_instances
        self.grid_size = grid_size
        self.n_quad_nodes = n_quad_nodes

    def run(self) -> GoldilocksReport:
        agents = {
            "random": RandomAgent(seed_offset=1),
            "greedy": GreedyAgent(),
            "optimal": OptimalAgent(),
        }
        difficulty_order = ["easy", "medium", "hard"]

        mean_j: dict[str, dict[str, float]] = {name: {} for name in agents}

        for level in difficulty_order:
            params = DIFFICULTY_LEVELS[level]

            config = GeneratorConfig(
                grid_size=self.grid_size,
                n_quad_nodes=self.n_quad_nodes,
                **params,
            )
            generator = RegimeSwitchingGenerator(config=config)

            # Collect raw J values for each agent
            j_values: dict[str, list[float]] = {name: [] for name in agents}

            for i in range(self.n_instances):
                problem = generator.sample(seed=i)

                for agent_name, agent in agents.items():
                    decisions = agent.decide(problem)
                    j = _compute_realized_utility(
                        decisions, problem.x_path, problem.lam,
                        problem.initial_regime, _linear_utility,
                    )
                    j_values[agent_name].append(j)

            for agent_name in agents:
                mean_j[agent_name][level] = float(np.mean(j_values[agent_name]))

        return GoldilocksReport(
            difficulty_levels=difficulty_order,
            agents=list(agents.keys()),
            mean_j=mean_j,
        )
