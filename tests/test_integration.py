"""End-to-end integration test: generate → solve → prompt → decide → score."""
import numpy as np
import pytest
from optimal_switching.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from optimal_switching.problems.regime_switching.verifier import RegimeSwitchingVerifier
from optimal_switching.problems.regime_switching.prompts import setup_prompt, step_prompt
from optimal_switching.agents.optimal_agent import OptimalAgent
from optimal_switching.agents.greedy_agent import GreedyAgent
from optimal_switching.agents.random_agent import RandomAgent


class TestEndToEnd:
    @pytest.fixture
    def env_components(self):
        config = GeneratorConfig(T_range=(5, 5), grid_size=100, n_quad_nodes=15)
        gen = RegimeSwitchingGenerator(config)
        verifier = RegimeSwitchingVerifier()
        return gen, verifier

    def test_full_pipeline(self, env_components):
        """Generate → prompt → simulate agent → score: full data flow."""
        gen, verifier = env_components
        problem = gen.sample(seed=42)

        # Step 1: Generate setup prompt
        setup = setup_prompt(problem)
        assert len(setup) > 0

        # Step 2: Simulate a multi-turn conversation using greedy agent
        agent = GreedyAgent()
        decisions = agent.decide(problem)

        completions = []
        prev_regime = problem.initial_regime
        for t in range(problem.T):
            user_msg = step_prompt(t, problem.z_path[t], prev_regime)
            assert "t=" in user_msg
            response = f"Reasoning here... s_{t} = {decisions[t]}"
            completions.append(response)
            prev_regime = int(decisions[t])

        # Step 3: Score
        score = verifier.score(completions, problem, mode="trajectory")
        assert isinstance(score, float)

    def test_reward_fn_interface(self, env_components):
        """Test the exact TRL reward_fn interface."""
        gen, verifier = env_components

        def reward_fn(completions_batch, problems_batch):
            return [
                verifier.score(c, p, mode="trajectory")
                for c, p in zip(completions_batch, problems_batch)
            ]

        problems = [gen.sample(seed=i) for i in range(5)]
        agent = OptimalAgent()
        completions_batch = []
        for p in problems:
            decisions = agent.decide(p)
            completions_batch.append(
                [f"s_{t} = {decisions[t]}" for t in range(p.T)]
            )

        scores = reward_fn(completions_batch, problems)
        assert len(scores) == 5
        for s in scores:
            assert isinstance(s, float)
            assert s >= 0.0  # optimal should score high

    def test_optimal_beats_greedy_beats_random(self, env_components):
        """Reward ordering: random < greedy < optimal over many instances."""
        gen, verifier = env_components
        agents = {
            "random": RandomAgent(seed_offset=1),
            "greedy": GreedyAgent(),
            "optimal": OptimalAgent(),
        }

        scores = {name: [] for name in agents}
        for seed in range(100):
            problem = gen.sample(seed=seed)
            for name, agent in agents.items():
                decisions = agent.decide(problem)
                completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]
                scores[name].append(verifier.score(completions, problem))

        mean_scores = {name: np.mean(s) for name, s in scores.items()}
        # Random < Greedy < Optimal on average
        assert mean_scores["random"] < mean_scores["greedy"]
        assert mean_scores["greedy"] < mean_scores["optimal"]
