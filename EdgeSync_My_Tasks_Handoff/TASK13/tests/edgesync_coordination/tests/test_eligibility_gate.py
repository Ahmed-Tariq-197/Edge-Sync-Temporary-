"""Unit tests for Hard Gate G(j) and Utility Function (Quality Gate 6).

Verifies:
1. Rejection of high CPU load (> 90%).
2. Rejection of low battery (< 15%).
3. Rejection of stale telemetry (> 2500 ms).
4. Feasibility check: t_exec + t_safety <= T_TTO_safe (budget = 1230 ms).
5. Utility engine weighting (w1=0.5, w2=0.3, w3=0.2).
"""
import pytest
from edgesync_coordination.decision.eligibility_gate import EligibilityGate, PeerTelemetry
from edgesync_coordination.decision.utility_engine import UtilityEngine


@pytest.fixture
def gate():
    return EligibilityGate(
        max_cpu_pct=90.0,
        min_battery_pct=15.0,
        max_telemetry_age_ms=2500.0,
        dt_exec_nominal_ms=866.0,
        dt_safety_ms=364.0
    )


@pytest.fixture
def utility_eng(gate):
    return UtilityEngine(eligibility_gate=gate)


def test_gate_rejection_criteria(gate):
    # Valid nominal peer
    good_peer = PeerTelemetry(cpu_load_pct=45.0, battery_pct=80.0, nav_status=1, last_heartbeat_age_ms=200.0)
    passed, _ = gate.evaluate_hard_gate(good_peer)
    assert passed is True

    # High CPU (> 90%)
    high_cpu_peer = PeerTelemetry(cpu_load_pct=92.0, battery_pct=80.0, nav_status=1, last_heartbeat_age_ms=200.0)
    passed, reason = gate.evaluate_hard_gate(high_cpu_peer)
    assert passed is False
    assert "CPU" in reason

    # Low battery (< 15%)
    low_bat_peer = PeerTelemetry(cpu_load_pct=40.0, battery_pct=12.0, nav_status=1, last_heartbeat_age_ms=200.0)
    passed, reason = gate.evaluate_hard_gate(low_bat_peer)
    assert passed is False
    assert "Battery" in reason

    # Stale telemetry (> 2500 ms)
    stale_peer = PeerTelemetry(cpu_load_pct=40.0, battery_pct=80.0, nav_status=1, last_heartbeat_age_ms=3000.0)
    passed, reason = gate.evaluate_hard_gate(stale_peer)
    assert passed is False
    assert "stale" in reason


def test_feasibility_boundary(gate):
    # Total required budget: 866ms + 364ms = 1230ms (1.23s)
    # TTO safe = 2.0s -> Feasible
    passed, _ = gate.evaluate_feasibility(tto_safe_s=2.0)
    assert passed is True

    # TTO safe = 1.0s (< 1.23s) -> Infeasible (Zone 3 Critical Boundary)
    passed, reason = gate.evaluate_feasibility(tto_safe_s=1.0)
    assert passed is False
    assert "budget" in reason


def test_utility_calculation(utility_eng):
    peer = PeerTelemetry(cpu_load_pct=30.0, battery_pct=90.0, nav_status=1, last_heartbeat_age_ms=100.0, rtt_ms=10.0)

    # Feasible handoff with 2.23s safe TTO (1.0s spare margin = full Phi_feas)
    score = utility_eng.compute_utility(peer, tto_safe_s=2.23, phi_fov=1.0, packet_loss_rate=0.0)

    # w1*1.0 + w2*1.0 + w3*0.9 = 0.5 + 0.3 + 0.18 = 0.98
    assert pytest.approx(score, abs=0.05) == 0.98
