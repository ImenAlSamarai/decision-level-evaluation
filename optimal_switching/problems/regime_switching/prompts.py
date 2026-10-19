"""Multi-turn prompt format for stateful conversation manager."""
from optimal_switching.problems.regime_switching.generator import RegimeSwitchingProblem


def setup_prompt(problem: RegimeSwitchingProblem) -> str:
    """Generate Turn 0 setup message.

    The prompt explicitly states the payoff structure so the model
    understands what each action means financially.

    Args:
        problem: RegimeSwitchingProblem instance with parameters and initial regime.

    Returns:
        Setup prompt string for the initial conversation turn.
    """
    ir_label = "ON" if problem.initial_regime == 1 else "OFF"
    return (
        f"You are managing a trading strategy over T={problem.T} steps.\n"
        f"\n"
        f"RULES:\n"
        f"- At each step t, you choose s_t = 1 (ON) or s_t = 0 (OFF).\n"
        f"- If ON (s_t=1): you earn the PnL for that step. "
        f"Expected PnL = {problem.alpha:.4f} x Z_t.\n"
        f"- If OFF (s_t=0): you earn nothing (PnL = 0).\n"
        f"- Every time you SWITCH (change from ON to OFF or OFF to ON), "
        f"you pay a cost of {problem.lam:.4f}.\n"
        f"- If you stay in the same state, no switching cost.\n"
        f"\n"
        f"YOUR GOAL: Maximize total profit = sum of PnL earned "
        f"minus switching costs paid.\n"
        f"\n"
        f"PARAMETERS:\n"
        f"  Signal strength: alpha = {problem.alpha:.4f}\n"
        f"  Switching cost:  lambda = {problem.lam:.4f}\n"
        f"  Starting state:  s_{{-1}} = {problem.initial_regime} ({ir_label})\n"
        f"  Horizon:         T = {problem.T} steps\n"
        f"\n"
        f"You will receive one signal observation Z_t at a time.\n"
        f"State your decision as: s_t = 0 or s_t = 1"
    )


def step_prompt(t: int, z_t: float, prev_regime: int) -> str:
    """Generate user message for a single time step.

    Args:
        t: Current time step (0-indexed).
        z_t: Signal value at time t.
        prev_regime: Previous regime s_{t-1} (or s_{-1} if t=0).

    Returns:
        Step prompt string with current observation and state.
    """
    state = "ON" if prev_regime == 1 else "OFF"
    return (
        f"t={t} | Z_t = {z_t:+.4f} | "
        f"You are currently {state} (s_prev = {prev_regime})"
    )
