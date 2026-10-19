"""Abstract base class for problem generators."""
from abc import ABC, abstractmethod
from optimal_switching.base.problem import BaseProblem


class BaseGenerator(ABC):
    """Base class for problem generators.

    Generators create fully-solved problem instances.
    The solver is merged into the generator — callers
    never receive an unsolved instance.
    """

    @abstractmethod
    def sample(self, seed: int) -> BaseProblem:
        """Generate a single fully-solved problem instance.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            A fully-solved problem instance with optimal policy computed.
        """
        ...
