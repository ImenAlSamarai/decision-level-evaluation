import pytest
import numpy as np
from optimal_switching.base.problem import BaseProblem
from optimal_switching.base.generator import BaseGenerator
from optimal_switching.base.verifier import BaseVerifier


def test_base_problem_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseProblem()


def test_base_generator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseGenerator()


def test_base_verifier_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseVerifier()


def test_base_generator_requires_sample_method():
    """Subclass must implement sample(seed) -> BaseProblem."""
    class BadGenerator(BaseGenerator):
        pass

    with pytest.raises(TypeError):
        BadGenerator()


def test_base_verifier_requires_score_method():
    """Subclass must implement score(completion, problem) -> float."""
    class BadVerifier(BaseVerifier):
        pass

    with pytest.raises(TypeError):
        BadVerifier()
