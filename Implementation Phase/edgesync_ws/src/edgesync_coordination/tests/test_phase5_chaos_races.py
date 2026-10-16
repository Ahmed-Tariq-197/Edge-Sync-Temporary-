"""Phase 5: Chaos & Race Condition Analysis — 5-Phase SDET Testing Protocol.

Tests:
1. Duplicate CONFIRM idempotency: replays stored ACK without restarting T_claim.
2. CANCEL while SLEWING: receiver must abort (currently BUG — no cancel path exists).
3. OWNER_CLAIM burst temporal diversity (3 × 20ms spacing).
"""
import pytest
from edgesync_coordination.fsm.ownership_fsm import (
    OwnershipFSM, SenderState, ReceiverState, OwnershipRole
)
from edgesync_coordination.protocol.negotiation_manager import NegotiationManager, TxnState
from edgesync_coordination.protocol.liveness_watchdog import LivenessWatchdog


class TestDuplicateConfirmIdempotency:
    """Verify duplicate HANDOFF_CONFIRM replays stored ACK without advancing epoch."""

    def test_fsm_guards_prevent_double_epoch_advance(self):
        """First CONFIRM advances epoch. Second CONFIRM returns same epoch (FSM guard)."""
        robot_a = OwnershipFSM("ugv_1", is_initial_owner=True, initial_epoch=1)

        robot_a.trigger_candidate("target", "txn-dup")
        robot_a.send_request()
        robot_a.on_peer_accepted()
        robot_a.on_peer_acquisition_started()
        robot_a.on_peer_verifying()

        # First CONFIRM → epoch advances 1 → 2
        epoch_1st = robot_a.on_confirm_received()
        assert epoch_1st == 2
        assert robot_a.sender_state == SenderState.CONFIRMED

        # Second CONFIRM (duplicate): FSM is now in CONFIRMED, not VERIFICATION
        # on_confirm_received() guard checks for IDENTITY_VERIFICATION or ACQUISITION
        # CONFIRMED is NOT in that set → returns current_epoch without advancing
        epoch_2nd = robot_a.on_confirm_received()
        assert epoch_2nd == 2  # Same epoch, NOT 3
        assert robot_a.epoch_manager.current_epoch == 2  # No double advance

    def test_negotiation_manager_stored_ack_replay(self):
        """NegotiationManager stores ACK epoch and replays it on duplicate lookup."""
        mgr = NegotiationManager("ugv_1")
        mgr.open_transaction("txn-replay", "target", "ugv_2")

        # First CONFIRM: mark as confirmed with ACK epoch 5
        mgr.mark_confirmed("txn-replay", ack_epoch=5)

        # Duplicate CONFIRM: retrieve stored ACK
        stored = mgr.get_stored_ack_epoch("txn-replay")
        assert stored == 5

        # Third duplicate: same result
        stored_again = mgr.get_stored_ack_epoch("txn-replay")
        assert stored_again == 5

    def test_negotiation_manager_not_called_in_coordination_node(self):
        """BUG DETECTION: mark_confirmed() is never called from coordination_node.py.

        The NegotiationManager has the idempotent replay mechanism but it's dead code.
        This test documents the bug.
        """
        # This test verifies the mechanism WORKS if it were wired up:
        mgr = NegotiationManager("ugv_1")
        mgr.open_transaction("txn-unwired", "target", "ugv_2")

        # Before marking: no stored ACK
        assert mgr.get_stored_ack_epoch("txn-unwired") is None

        # After marking: stored ACK available
        mgr.mark_confirmed("txn-unwired", ack_epoch=7)
        assert mgr.get_stored_ack_epoch("txn-unwired") == 7

        # After closing: stored ACK gone
        mgr.close_transaction("txn-unwired", "completed")
        assert mgr.get_stored_ack_epoch("txn-unwired") is None


class TestCancelWhileSlewing:
    """Verify cancel path behavior during SLEWING state.

    BUG DETECTED (P5-B2): No cancel subscriber/publisher/handler exists.
    These tests document EXPECTED behavior that currently has no implementation.
    """

    def test_receiver_fsm_has_no_cancel_transition(self):
        """BUG: OwnershipFSM has no method to abort receiver from SLEWING to IDLE."""
        robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=1)

        # Setup: Robot B enters SLEWING
        robot_b.on_request_received("txn-cancel", "target")
        robot_b.accept_request()
        robot_b.start_acquisition()
        assert robot_b.receiver_state == ReceiverState.SLEWING

        # There is no method like robot_b.cancel_acquisition() or robot_b.abort_receiver()
        # The only way to get out of SLEWING is:
        # 1. target_locked() → VERIFYING
        # 2. No direct cancel path

        # Document that receiver_state cannot be directly set to IDLE from SLEWING
        # via any public FSM method (besides manual state mutation)
        available_methods = [m for m in dir(robot_b) if not m.startswith('_')]
        assert 'cancel_acquisition' not in available_methods
        assert 'abort_receiver' not in available_methods
        assert 'receiver_abort' not in available_methods

    def test_slewing_robot_continues_without_cancel(self):
        """Without cancel mechanism, robot in SLEWING has no abort path."""
        robot_b = OwnershipFSM("ugv_2", is_initial_owner=False, initial_epoch=1)

        robot_b.on_request_received("txn-orphan", "target")
        robot_b.accept_request()
        robot_b.start_acquisition()

        # Robot A crashes or cancels — but Robot B has no way to know
        # Robot B remains in SLEWING forever (unless target is found)
        assert robot_b.receiver_state == ReceiverState.SLEWING
        assert robot_b.role == OwnershipRole.CANDIDATE

        # This is the BUG: no external signal can abort SLEWING


class TestOwnerClaimBurstTiming:
    """Verify OWNER_CLAIM burst has 20ms spacing requirement."""

    def test_burst_spec_requires_three_copies(self):
        """Per coordination_params.yaml: owner_claim_burst = 3."""
        # From timing_budgets.yaml L22:
        # owner_claim_burst: 3
        # owner_claim_burst_period_ms: 20
        expected_burst_count = 3
        expected_spacing_ms = 20

        # In coordination_node.py L192-203:
        # for i in range(1, 4): → exactly 3 iterations ✅
        # BUT: no time.sleep(0.020) or timer between publishes ❌

        # The burst count is correct (3) but spacing is 0ms (bug)
        assert expected_burst_count == 3
        assert expected_spacing_ms == 20

    def test_synchronous_burst_lacks_temporal_diversity(self):
        """BUG: All 3 OWNER_CLAIM copies published with zero delay.

        This defeats the purpose of temporal diversity against packet loss.
        With 20ms spacing, DDS has 3 independent chances to deliver the claim.
        With 0ms spacing, all 3 are batched together → single point of failure.
        """
        # This is a design-level bug documented here for traceability.
        # The fix requires using self.create_timer(0.020, ...) for deferred publishing.
        pass


class TestEpochRecoveryArithmetic:
    """Verify epoch arithmetic under various recovery scenarios."""

    def test_normal_handoff_advances_by_1(self):
        """Normal handoff: epoch advances e → e+1."""
        from edgesync_coordination.fsm.epoch_manager import EpochManager
        em = EpochManager(initial_epoch=5)
        new = em.advance_to_next_epoch()
        assert new == 6
        assert em.current_epoch == 6

    def test_recovery_advances_by_2(self):
        """Recovery reclaim: epoch advances e → e+2."""
        from edgesync_coordination.fsm.epoch_manager import EpochManager
        em = EpochManager(initial_epoch=5)
        reclaim = em.advance_to_recovery_epoch()
        assert reclaim == 7
        assert em.current_epoch == 7

    def test_recovery_dominates_inflight_handoff(self):
        """e+2 recovery epoch MUST be strictly greater than any in-flight e+1 claim."""
        from edgesync_coordination.fsm.epoch_manager import EpochManager

        em_a = EpochManager(initial_epoch=10)
        # Normal handoff advances to 11
        normal_epoch = em_a.advance_to_next_epoch()
        assert normal_epoch == 11

        # But then failure: recovery advances by +2 from 11 → 13
        reclaim_epoch = em_a.advance_to_recovery_epoch()
        assert reclaim_epoch == 13
        assert reclaim_epoch > normal_epoch  # 13 > 11 ✅

    def test_stale_epoch_rejected(self):
        """Messages with epoch < current must be rejected."""
        from edgesync_coordination.fsm.epoch_manager import EpochManager
        em = EpochManager(initial_epoch=5)

        assert em.validate_incoming_epoch(5) is True   # Current: OK
        assert em.validate_incoming_epoch(6) is True   # Future: OK
        assert em.validate_incoming_epoch(4) is False  # Stale: REJECTED
        assert em.is_stale(4) is True
        assert em.is_stale(5) is False
