"""Unit tests for Scenario Orchestrator and Wilson Interval (Quality Gate 10).

Verifies:
1. Wilson score confidence interval calculation formula.
2. Trial execution across all 7 scenarios (A through G).
3. Zero unbounded dual-ownership assertions.
4. Final Markdown report formatting.
"""
import pytest
from edgesync_observability.scripts.scenario_orchestrator import (
    ScenarioOrchestrator, compute_wilson_interval
)


def test_wilson_confidence_interval_math():
    # 50 trials, 47 successes (~94%)
    low, high = compute_wilson_interval(47, 50, confidence=0.95)

    assert low > 80.0
    assert high <= 100.0
    assert low < 94.0 < high

    # Edge cases
    assert compute_wilson_interval(0, 0) == (0.0, 0.0)
    assert compute_wilson_interval(10, 10)[0] > 70.0


def test_orchestrator_runs_all_seven_scenarios():
    orch = ScenarioOrchestrator(output_dir="/tmp/test_edgesync_reports")

    # Run 1 trial for each scenario (A through G)
    for code in "ABCDEFG":
        res = orch.run_scenario_trial(code, trial_idx=0, subject_id="Test_Subject_1")
        assert res["scenario"] == code
        assert res["success"] is True
        assert res["dual_ownership_unbounded"] is False
        assert res["overlap_ms"] <= 200.0  # Bounded by T_claim

    report = orch.generate_final_report()
    assert "Handoff Success Rate (HSR)" in report
    assert "Unbounded Dual-Ownership" in report
    assert "Scenario A" in report
    assert "Scenario G" in report
