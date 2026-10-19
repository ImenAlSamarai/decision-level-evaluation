"""Abstract base class for problem instances."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseProblem(ABC):
    """Base class for problem instances.

    Every concrete problem must define its own fields.
    The generator always returns fully-solved instances.
    """

    @abstractmethod
    def horizon(self) -> int:
        """Return the number of decision steps T."""
        ...
