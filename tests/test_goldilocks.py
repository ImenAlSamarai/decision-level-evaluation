"""Tests for the Goldilocks validation suite."""
from __future__ import annotations

import pytest

from optimal_switching.validation.goldilocks import GoldilocksValidator, GoldilocksReport

# Use production-quality grid with enough instances for statistical stability.
VALIDATOR = GoldilocksValidator(n_instances=200, grid_size=200, n_quad_nodes=20)


@pytest.fixture(scope="module")
def report() -> GoldilocksReport:
    return VALIDATOR.run()


def test_report_structure(report: GoldilocksReport) -> None:
    assert len(report.difficulty_levels) == 3
    assert set(report.difficulty_levels) == {"easy", "medium", "hard"}
    assert len(report.agents) == 3
    assert set(report.agents) == {"random", "greedy", "optimal"}


def test_optimal_beats_greedy(report: GoldilocksReport) -> None:
    """Mean J(optimal) > Mean J(greedy) at all levels."""
    for level in report.difficulty_levels:
        assert report.mean_j["optimal"][level] > report.mean_j["greedy"][level], (
            f"Optimal should beat greedy at {level}"
        )


def test_greedy_beats_random(report: GoldilocksReport) -> None:
    """Mean J(greedy) > Mean J(random) at all levels."""
    for level in report.difficulty_levels:
        assert report.mean_j["greedy"][level] > report.mean_j["random"][level], (
            f"Greedy should beat random at {level}"
        )


def test_greedy_capture_decreases(report: GoldilocksReport) -> None:
    """Greedy captures less of optimal at harder levels (planning matters more)."""
    captures = report.greedy_capture_pct()
    assert captures["easy"] > captures["medium"] > captures["hard"], (
        f"Expected decreasing: easy={captures['easy']:.1f}% > "
        f"medium={captures['medium']:.1f}% > hard={captures['hard']:.1f}%"
    )


def test_pass_criteria(report: GoldilocksReport) -> None:
    assert report.all_pass(), "Expected all_pass() to return True"


def test_report_string(report: GoldilocksReport) -> None:
    s = str(report)
    assert "Goldilocks" in s
    assert "PASS" in s or "FAIL" in s
    assert "capture" in s.lower()
