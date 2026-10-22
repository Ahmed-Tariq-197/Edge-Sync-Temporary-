"""Unit tests for Ownership FSM & Zero Dual-Ownership Invariant (Quality Gate 6).

Verifies:
1. Robot A remains sole owner throughout COMMITTED, ACQUISITION, and VERIFICATION.
2. Ownership transfers to Robot B strictly upon receiving HANDOFF_CONFIRM_ACK (epoch e+1).
3. Failure abort triggers RECOVERY and preempts in-flight claims at epoch e+2.
"""
import pytest
from edgesync_coordination.fsm.ownership_fsm import (
    OwnershipFSM, SenderState, ReceiverState, OwnershipRole
)


def test_nominal_handoff_fsm_flow():
    robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=1)
    robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=1)

    # Initial state
    assert robot_a.role == OwnershipRole.OWNER
    assert robot_a.sender_state == SenderState.TRACKING
    assert robot_b.role == OwnershipRole.STANDBY
    assert robot_b.receiver_state == ReceiverState.IDLE

    # 1. Trigger handoff candidate on Robot A
    assert robot_a.trigger_candidate("target_human", "txn-101") is True
    assert robot_a.send_request() is True
    assert robot_a.sender_state == SenderState.NEGOTIATING

    # 2. Robot B receives request and accepts
    assert robot_b.on_request_received("txn-101", "target_human") is True
    assert robot_b.accept_request() is True
    assert robot_b.role == OwnershipRole.CANDIDATE
    assert robot_b.receiver_state == ReceiverState.ACCEPTED

    # 3. Robot A commits (Zero Dual-Ownership Invariant check!)
    assert robot_a.on_peer_accepted() is True
    assert robot_a.sender_state == SenderState.COMMITTED
    # ROBOT A MUST REMAIN SOLE AUTHORITATIVE OWNER!
    assert robot_a.role == OwnershipRole.OWNER
    assert robot_b.role == OwnershipRole.CANDIDATE

    # 4. Acquisition & Verification
    assert robot_b.start_acquisition() is True
    assert robot_a.on_peer_acquisition_started() is True
    assert robot_b.target_locked() is True
    assert robot_a.on_peer_verifying() is True

    # 5. Confirmation & Epoch increment
    assert robot_b.send_confirm() is True
    assert robot_b.receiver_state == ReceiverState.AWAITING_ACK

    # Robot A confirms and generates epoch e+1
    new_epoch = robot_a.on_confirm_received()
    assert new_epoch == 2
    assert robot_a.sender_state == SenderState.CONFIRMED

    # 6. Robot B receives ACK and BECOMES OWNER!
    assert robot_b.on_confirm_ack_received(new_epoch) is True
    assert robot_b.receiver_state == ReceiverState.TRACKING
    assert robot_b.role == OwnershipRole.OWNER
    assert robot_b.epoch_manager.current_epoch == 2

    # 7. Robot A receives OWNER_CLAIM burst and demotes to STANDBY
    assert robot_a.on_owner_claim_received() is True
    assert robot_a.sender_state == SenderState.STANDBY
    assert robot_a.role == OwnershipRole.STANDBY


def test_watchdog_failure_and_epoch_e_plus_2_recovery():
    robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=1)

    robot_a.trigger_candidate("target_human", "txn-102")
    robot_a.send_request()
    robot_a.on_peer_accepted()
    robot_a.on_peer_acquisition_started()

    # Peer crashes during acquisition -> Watchdog triggers abort
    reclaim_epoch = robot_a.abort_to_recovery("Watchdog timeout (ERR_CANDIDATE_DEAD)")

    assert robot_a.sender_state == SenderState.RECOVERY
    assert robot_a.role == OwnershipRole.OWNER
    # Must reclaim at epoch e+2 (1 + 2 = 3) to strictly dominate any stale e+1 claims!
    assert reclaim_epoch == 3
    assert robot_a.epoch_manager.current_epoch == 3

    # Resume solo tracking
    robot_a.recovery_to_solo_tracking()
    assert robot_a.sender_state == SenderState.TRACKING
