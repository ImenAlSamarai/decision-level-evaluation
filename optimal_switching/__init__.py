"""Optimal-switching environment with an exact Bellman reference, used in the paper (problems/regime_switching)."""
from optimal_switching.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    RegimeSwitchingProblem,
)
from optimal_switching.problems.regime_switching.verifier import RegimeSwitchingVerifier
from optimal_switching.problems.regime_switching.prompts import setup_prompt, step_prompt
from optimal_switching.agents.random_agent import RandomAgent
from optimal_switching.agents.greedy_agent import GreedyAgent
from optimal_switching.agents.optimal_agent import OptimalAgent
from optimal_switching.validation.goldilocks import GoldilocksValidator

__all__ = [
    "GeneratorConfig",
    "RegimeSwitchingGenerator",
    "RegimeSwitchingProblem",
    "RegimeSwitchingVerifier",
    "setup_prompt",
    "step_prompt",
    "RandomAgent",
    "GreedyAgent",
    "OptimalAgent",
    "GoldilocksValidator",
]
