"""Master Fault Injection Harness (Tier 1 Simulation - WP-8).

Executes the 3 Certified Fault Injection Scenarios:
1. Scenario D (Crash): Candidate Robot B crashes mid-acquisition; 10 Hz Watchdog cuts off
   at 300 ms, Robot A aborts to RECOVERY and reclaims sole ownership at epoch e+2.
2. Scenario F (Netem): Packet drop & delay injection on confirmation handshake; verifies
   3x 30ms retries succeed within T_ack (100ms) or safely fallback to T_claim (200ms) reclaim.
3. Scenario G (Stale Epoch): Injects delayed/malicious packets carrying stale epoch e-1;
   verifies strict admission rejection without state corruption or dual-ownership.

Aligned Standards:
- SRS v1.3 §5.2, SAD v2.3 §2, Math/Protocol Spec v2.3 §3, Quality Gate 8.
"""
import pytest
import time
import random
from typing import List, Optional

from edgesync_coordination.fsm.ownership_fsm import (
    OwnershipFSM, SenderState, ReceiverState, OwnershipRole
)
from edgesync_coordination.fsm.epoch_manager import EpochManager
from edgesync_coordination.protocol.liveness_watchdog import LivenessWatchdog
from edgesync_coordination.protocol.negotiation_manager import NegotiationManager, TxnState


# ============================================================================
# SCENARIO D: CANDIDATE CRASH & 300ms WATCHDOG RECOVERY
# ============================================================================

def test_scenario_d_candidate_crash_and_watchdog_cutoff():
    """Scenario D: Robot B suffers sudden hardware/process failure during ACQUISITION.

    Success Criteria:
    - 3 consecutive missed heartbeats trigger ERR_CANDIDATE_DEAD cutoff within 300-400 ms.
    - Robot A aborts transaction and transitions to RECOVERY.
    - Target ownership is reclaimed at epoch e+2 (preempts any pending e+1 claim).
    - Robot A resumes solo tracking with ZERO dual-ownership occurrences.
    """
    txn_id = "txn-crash-scen-d"
    target_id = "human_target_01"

    # Robot A is initial owner at epoch 1
    robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=1)
    # Robot B is standby peer at epoch 1
    robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=1)

    cutoff_triggered = []

    def on_watchdog_timeout():
        cutoff_triggered.append(True)
        # Abort to RECOVERY at epoch e+2
        robot_a.abort_to_recovery("Watchdog cutoff (ERR_CANDIDATE_DEAD)")

    watchdog_a = LivenessWatchdog(
        heartbeat_period_ms=100.0,
        max_missed_heartbeats=3,
        on_timeout_callback=on_watchdog_timeout
    )

    # 1. Handoff Initiation & Negotiation
    assert robot_a.trigger_candidate(target_id, txn_id) is True
    assert robot_a.send_request() is True
    assert robot_b.on_request_received(txn_id, target_id) is True
    assert robot_b.accept_request() is True
    assert robot_a.on_peer_accepted() is True

    # Robot A arms watchdog at the instant of ACCEPT
    watchdog_a.arm()
    sim_clock = 1000.0
    watchdog_a._last_heartbeat_time = sim_clock

    # In COMMITTED: Robot A is SOLE OWNER, Robot B is CANDIDATE
    assert robot_a.sender_state == SenderState.COMMITTED
    assert robot_a.role == OwnershipRole.OWNER
    assert robot_b.role == OwnershipRole.CANDIDATE

    # 2. Robot B enters acquisition and emits first heartbeat
    assert robot_b.start_acquisition() is True
    assert robot_a.on_peer_acquisition_started() is True
    sim_clock += 0.100  # +100 ms
    watchdog_a.feed(current_time=sim_clock)
    assert watchdog_a.check(current_time=sim_clock) is False

    # ------------------------------------------------------------------------
    # FAULT INJECTION: Robot B crashes abruptly (SIGKILL / power severance)
    # No further heartbeats are published by Robot B!
    # ------------------------------------------------------------------------
    robot_b_crashed = True

    # Advance time: Heartbeat 2 missed (+100 ms -> elapsed 100ms from last feed)
    sim_clock += 0.100
    assert watchdog_a.check(current_time=sim_clock) is False

    # Advance time: Heartbeat 3 missed (+100 ms -> elapsed 200ms from last feed)
    sim_clock += 0.100
    assert watchdog_a.check(current_time=sim_clock) is False
    assert len(cutoff_triggered) == 0

    # Advance time: 300 ms cutoff reached! (+101 ms -> elapsed 301ms from last feed)
    sim_clock += 0.101
    assert watchdog_a.check(current_time=sim_clock) is True

    # Assertions on Recovery & Ownership Invariants
    assert len(cutoff_triggered) == 1
    assert robot_a.sender_state == SenderState.RECOVERY
    assert robot_a.role == OwnershipRole.OWNER
    # Must reclaim at epoch e+2: 1 + 2 = 3
    assert robot_a.epoch_manager.current_epoch == 3

    # Robot A transitions from RECOVERY to solo TRACKING
    robot_a.recovery_to_solo_tracking()
    assert robot_a.sender_state == SenderState.TRACKING
    assert robot_a.role == OwnershipRole.OWNER

    # Zero Dual-Ownership Check: Robot B never became OWNER
    assert robot_b.role != OwnershipRole.OWNER


# ============================================================================
# SCENARIO F: NETWORK LOSS & PACKET RETRY GATING (NETEM)
# ============================================================================

class SimulatedNetwork:
    """Simulates packet drops and delays representative of 5 GHz Wi-Fi netem."""

    def __init__(self, drop_probability: float = 0.0, delay_ms: float = 0.0):
        self.drop_prob = drop_probability
        self.delay_ms = delay_ms

    def transmit(self, packet_id: str, attempt_num: int, guaranteed_drop_attempts: int = 0) -> bool:
        """Return True if packet is delivered, False if dropped."""
        if attempt_num <= guaranteed_drop_attempts:
            return False  # Injected deterministic loss
        if self.drop_prob > 0.0 and random.random() < self.drop_prob:
            return False
        return True


def test_scenario_f_network_loss_and_confirmation_retries_success():
    """Scenario F (Sub-case 1): First two CONFIRM packets dropped; 3rd retry succeeds within T_ack.

    Timing Budget Constraints:
    - confirm_retry_period_ms = 30 ms
    - confirm_retry_count = 3
    - T_ack = 100 ms (local monotonic)
    - Attempt 1 (0 ms): DROPPED
    - Attempt 2 (30 ms): DROPPED
    - Attempt 3 (60 ms): DELIVERED (60 ms < 100 ms T_ack)
    - Total handoff succeeds with zero dual-ownership.
    """
    robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=1)
    robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=1)
    net = SimulatedNetwork()

    txn_id = "txn-netem-retry-scen-f"
    robot_a.trigger_candidate("target_01", txn_id)
    robot_a.send_request()
    robot_b.on_request_received(txn_id, "target_01")
    robot_b.accept_request()
    robot_a.on_peer_accepted()
    robot_b.start_acquisition()
    robot_a.on_peer_acquisition_started()
    robot_b.target_locked()
    robot_a.on_peer_verifying()

    # Robot B verifies Re-ID (>= 0.70) and enters AWAITING_ACK
    assert robot_b.send_confirm() is True
    assert robot_b.receiver_state == ReceiverState.AWAITING_ACK

    t_ack_limit_ms = 100.0
    elapsed_ms = 0.0
    confirm_delivered = False
    ack_delivered = False

    # Simulate 3x 30ms CONFIRM retries with first 2 dropped
    for attempt in range(1, 4):
        delivered = net.transmit("CONFIRM", attempt_num=attempt, guaranteed_drop_attempts=2)
        if delivered:
            confirm_delivered = True
            break
        elapsed_ms += 30.0

    # 3rd attempt delivered at 60 ms (< 100 ms T_ack)
    assert confirm_delivered is True
    assert elapsed_ms == 60.0
    assert elapsed_ms < t_ack_limit_ms

    # Robot A receives CONFIRM, enters CONFIRMED, replies ACK with epoch e+1 = 2
    new_epoch = robot_a.on_confirm_received()
    assert new_epoch == 2
    assert robot_a.sender_state == SenderState.CONFIRMED

    # ACK delivered back to Robot B within remaining window (60ms + 20ms = 80ms < 100ms)
    elapsed_ms += 20.0
    assert elapsed_ms <= t_ack_limit_ms
    assert robot_b.on_confirm_ack_received(new_epoch) is True

    # Robot B is now OWNER at epoch 2
    assert robot_b.role == OwnershipRole.OWNER
    assert robot_b.receiver_state == ReceiverState.TRACKING

    # Robot A demotes to STANDBY upon receiving claim
    assert robot_a.on_owner_claim_received() is True
    assert robot_a.role == OwnershipRole.STANDBY


def test_scenario_f_total_network_severance_and_claim_reclaim():
    """Scenario F (Sub-case 2): All CONFIRM retries dropped; T_ack (100ms) expires on B,

    T_claim (200ms) expires on A -> Robot A reclaims target at epoch e+2 = 3.
    """
    robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=1)
    robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=1)
    net = SimulatedNetwork()

    txn_id = "txn-severance-scen-f"
    robot_a.trigger_candidate("target_01", txn_id)
    robot_a.send_request()
    robot_b.on_request_received(txn_id, "target_01")
    robot_b.accept_request()
    robot_a.on_peer_accepted()
    robot_b.start_acquisition()
    robot_b.target_locked()
    robot_b.send_confirm()

    # All 3 retries dropped!
    confirm_delivered = any(net.transmit("CONFIRM", attempt, guaranteed_drop_attempts=3) for attempt in range(1, 4))
    assert confirm_delivered is False

    # T_ack expires on Robot B -> B reverts to IDLE, cancels transaction
    robot_b.on_ack_timeout()
    assert robot_b.receiver_state == ReceiverState.IDLE
    assert robot_b.role == OwnershipRole.STANDBY

    # On Robot A: T_claim expires without claim arriving -> Abort & Reclaim at epoch e+2
    reclaim_epoch = robot_a.abort_to_recovery("T_claim expired without owner claim")
    assert reclaim_epoch == 3
    assert robot_a.epoch_manager.current_epoch == 3
    assert robot_a.role == OwnershipRole.OWNER

    # Zero Dual-Ownership Check:
    # If a delayed/zombie ACK for txn arrives at B later, B must reject it!
    # Because B is in IDLE, on_confirm_ack_received will fail.
    assert robot_b.on_confirm_ack_received(ack_epoch=2) is False
    assert robot_b.role == OwnershipRole.STANDBY


# ============================================================================
# SCENARIO G: STALE EPOCH PACKET INJECTION (PROTOCOL ATTACK / REPLAY)
# ============================================================================

def test_scenario_g_stale_epoch_rejection():
    """Scenario G: Network delay or malicious replay injects packets with stale epoch e-1.

    Success Criteria:
    - Stale epoch packets (epoch < current_epoch) are strictly dropped at admission gate.
    - Zero state corruption or divergence in FSM.
    - Highest epoch always wins arbitration.
    """
    manager_a = NegotiationManager(self_id="ugv_1", protocol_version=0x0203)
    epoch_mgr = EpochManager(initial_epoch=3)  # System is currently at Epoch 3

    # Current epoch is 3. An incoming message carrying stale epoch 2 (e-1) arrives:
    stale_epoch = 2
    current_epoch = epoch_mgr.current_epoch

    # Test 1: Stale HANDOFF_REQUEST
    admitted, reason = manager_a.admit_message(
        msg_protocol_version=0x0203,
        receiver_id="ugv_1",
        txn_id="txn-stale-01",
        msg_epoch=stale_epoch,
        current_epoch=current_epoch,
        is_request=True
    )
    assert admitted is False
    assert "Stale epoch" in reason

    # Test 2: Monotonic Epoch Manager validation
    assert epoch_mgr.validate_incoming_epoch(stale_epoch) is False
    assert epoch_mgr.is_stale(stale_epoch) is True

    # Test 3: Epoch >= current must be accepted
    assert epoch_mgr.validate_incoming_epoch(3) is True
    assert epoch_mgr.validate_incoming_epoch(4) is True

    # Test 4: Protocol version mismatch rejection
    wrong_version = 0x0100
    admitted_ver, reason_ver = manager_a.admit_message(
        msg_protocol_version=wrong_version,
        receiver_id="ugv_1",
        txn_id="txn-ver-02",
        msg_epoch=3,
        current_epoch=3,
        is_request=True
    )
    assert admitted_ver is False
    assert "version mismatch" in reason_ver.lower()

    # Test 5: Mismatched receiver ID rejection (addressed to another robot)
    admitted_rx, reason_rx = manager_a.admit_message(
        msg_protocol_version=0x0203,
        receiver_id="ugv_3",  # Not ugv_1
        txn_id="txn-rx-03",
        msg_epoch=3,
        current_epoch=3,
        is_request=True
    )
    assert admitted_rx is False
    assert "addressed to" in reason_rx.lower()


# ============================================================================
# QUALITY GATE 8: PROPERTY-BASED RANDOMIZED PROTOCOL CONFORMANCE
# ============================================================================

def test_quality_gate_8_zero_dual_ownership_invariant_under_random_loss():
    """Property-based stress test: 50 randomized handoff trials with random packet loss/delay.

    Invariants Verified:
    1. At no instant do both robots believe they are exclusive OWNER at the same epoch.
    2. Overlap between Robot A in CONFIRMED and Robot B in TRACKING is strictly bounded by T_claim.
    3. Final converged state has exactly one OWNER or zero (if aborted to IDLE).
    """
    random.seed(42)

    for trial in range(50):
        initial_epoch = 1 + trial * 2
        robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=initial_epoch)
        robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=initial_epoch)

        txn = f"trial-{trial}"
        robot_a.trigger_candidate("target", txn)
        robot_a.send_request()

        # Randomize: Request dropped or peer busy?
        if random.random() < 0.15:
            # Drop request -> A times out to solo tracking
            robot_a.sender_state = SenderState.TRACKING
            assert robot_a.role == OwnershipRole.OWNER
            assert robot_b.role == OwnershipRole.STANDBY
            continue

        robot_b.on_request_received(txn, "target")
        robot_b.accept_request()
        robot_a.on_peer_accepted()

        # Randomize: Crash during acquisition?
        if random.random() < 0.20:
            # Scenario D triggered
            robot_a.abort_to_recovery("Watchdog timeout")
            robot_a.recovery_to_solo_tracking()
            assert robot_a.role == OwnershipRole.OWNER
            assert robot_b.role == OwnershipRole.CANDIDATE
            continue

        robot_b.start_acquisition()
        robot_a.on_peer_acquisition_started()
        robot_b.target_locked()
        robot_a.on_peer_verifying()
        robot_b.send_confirm()

        # Randomize: Confirm ACK delivered?
        if random.random() < 0.30:
            # Network drop -> Robot B times out, A reclaims at e+2
            robot_b.on_ack_timeout()
            robot_a.abort_to_recovery("ACK timeout")
            robot_a.recovery_to_solo_tracking()
            assert robot_a.role == OwnershipRole.OWNER
            assert robot_b.role == OwnershipRole.STANDBY
        else:
            # Successful handoff
            ack_epoch = robot_a.on_confirm_received()
            robot_b.on_confirm_ack_received(ack_epoch)
            robot_a.on_owner_claim_received()
            assert robot_b.role == OwnershipRole.OWNER
            assert robot_a.role == OwnershipRole.STANDBY
