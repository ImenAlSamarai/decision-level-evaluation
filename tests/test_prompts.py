"""Tests for multi-turn prompt generation."""
import pytest
from optimal_switching.problems.regime_switching.generator import GeneratorConfig, RegimeSwitchingGenerator
from optimal_switching.problems.regime_switching.prompts import setup_prompt, step_prompt


@pytest.fixture
def problem():
    config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
    gen = RegimeSwitchingGenerator(config)
    return gen.sample(seed=42)


class TestSetupPrompt:
    def test_contains_parameters(self, problem):
        prompt = setup_prompt(problem)
        assert str(problem.T) in prompt
        assert str(round(problem.lam, 4)) in prompt or f"{problem.lam:.4f}" in prompt
        assert str(round(problem.alpha, 4)) in prompt or f"{problem.alpha:.4f}" in prompt

    def test_contains_formula(self, problem):
        prompt = setup_prompt(problem)
        assert "E[X" in prompt or "Expected PnL" in prompt

    def test_contains_initial_regime(self, problem):
        prompt = setup_prompt(problem)
        assert str(problem.initial_regime) in prompt


class TestStepPrompt:
    def test_contains_time_and_signal(self, problem):
        prompt = step_prompt(t=0, z_t=0.42, prev_regime=0)
        assert "t=0" in prompt
        assert "0.42" in prompt

    def test_contains_previous_regime(self):
        prompt = step_prompt(t=3, z_t=-0.12, prev_regime=1)
        assert "1" in prompt

    def test_step_prompts_differ(self):
        p1 = step_prompt(t=0, z_t=0.5, prev_regime=0)
        p2 = step_prompt(t=1, z_t=-0.3, prev_regime=1)
        assert p1 != p2
