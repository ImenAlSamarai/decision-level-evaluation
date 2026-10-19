"""Abstract base class for verifiers."""
from abc import ABC, abstractmethod
from optimal_switching.base.problem import BaseProblem


class BaseVerifier(ABC):
    """Base class for verifiers.

    Verifiers parse LLM text completions, extract decisions,
    and return a float reward score.
    """

    @abstractmethod
    def score(self, completion: list[str], problem: BaseProblem,
              mode: str = "trajectory") -> float:
        """Score a model's completion against the optimal policy.

        Args:
            completion: List of assistant response strings, one per turn.
            problem: The fully-solved problem instance.
            mode: "trajectory" (training reward) or "per_step" (diagnostic).

        Returns:
            Float score. 1.0 = optimal, 0.0 = random-level.
        """
        ...
