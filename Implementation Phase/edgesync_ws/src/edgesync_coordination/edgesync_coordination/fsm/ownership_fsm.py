"""Explicit Ownership Finite State Machine (SAD v2.3 §2, Math/Protocol Spec v2.3 §3).

Strictly enforces the 5 System Invariants:
1. Zero Dual-Ownership: COMMITTED preserves Robot A as sole owner;
   ownership transfers strictly upon atomic exchange of HANDOFF_CONFIRM_ACK with epoch e+1.
2. Anti-deadlock: Watchdog timeout immediately transitions to RECOVERY.
"""
from enum import Enum, auto
from typing import Optional, Callable
from .epoch_manager import EpochManager


class SenderState(Enum):
    TRACKING = auto()
    HANDOFF_CANDIDATE = auto()
    NEGOTIATING = auto()
    COMMITTED = auto()
    ACQUISITION = auto()
    IDENTITY_VERIFICATION = auto()
    CONFIRMED = auto()
    STANDBY = auto()
    RECOVERY = auto()
    TERMINAL_FAILURE = auto()


class ReceiverState(Enum):
    IDLE = auto()
    EVALUATING = auto()
    ACCEPTED = auto()
    SLEWING = auto()
    SEARCHING = auto()
    VERIFYING = auto()
    AWAITING_ACK = auto()
    TRACKING = auto()


class OwnershipRole(Enum):
    OWNER = 1
    CANDIDATE = 2
    STANDBY = 3


class OwnershipFSM:
    """Finite State Machine orchestrating target ownership lifecycle and handoff arbitration."""

    def __init__(self, robot_id: str, is_initial_owner: bool = False, initial_epoch: int = 1):
        self.robot_id = robot_id
        self.epoch_manager = EpochManager(initial_epoch=initial_epoch)

        if is_initial_owner:
            self.role = OwnershipRole.OWNER
            self.sender_state = SenderState.TRACKING
            self.receiver_state = ReceiverState.IDLE
        else:
            self.role = OwnershipRole.STANDBY
            self.sender_state = SenderState.STANDBY
            self.receiver_state = ReceiverState.IDLE

        self.active_transaction_id: Optional[str] = None
        self.active_target_id: Optional[str] = None
        self.on_state_changed: Optional[Callable[[str, str], None]] = None

    # --- SENDER (ROBOT A) TRANSITIONS ---
    def trigger_candidate(self, target_id: str, txn_id: str) -> bool:
        """Zone 2 entered: TTO between 1.23s and 2.50s."""
        if self.role == OwnershipRole.OWNER and self.sender_state == SenderState.TRACKING:
            self.sender_state = SenderState.HANDOFF_CANDIDATE
            self.active_target_id = target_id
            self.active_transaction_id = txn_id
            self._notify_change("TRACKING -> HANDOFF_CANDIDATE")
            return True
        return False

    def send_request(self) -> bool:
        """P2P HANDOFF_REQUEST sent to peer candidate."""
        if self.sender_state == SenderState.HANDOFF_CANDIDATE:
            self.sender_state = SenderState.NEGOTIATING
            self._notify_change("HANDOFF_CANDIDATE -> NEGOTIATING")
            return True
        return False

    def on_peer_accepted(self) -> bool:
        """Peer replied ACCEPT: commit transaction (Sender remains sole owner)."""
        if self.sender_state == SenderState.NEGOTIATING:
            self.sender_state = SenderState.COMMITTED
            self._notify_change("NEGOTIATING -> COMMITTED (A remains SOLE OWNER)")
            return True
        return False

    def on_peer_acquisition_started(self) -> bool:
        """Peer began slewing/acquisition."""
        if self.sender_state == SenderState.COMMITTED:
            self.sender_state = SenderState.ACQUISITION
            self._notify_change("COMMITTED -> ACQUISITION")
            return True
        return False

    def on_peer_verifying(self) -> bool:
        """Peer acquired target centroid and requested Re-ID verification."""
        if self.sender_state in (SenderState.COMMITTED, SenderState.ACQUISITION):
            self.sender_state = SenderState.IDENTITY_VERIFICATION
            self._notify_change("ACQUISITION -> IDENTITY_VERIFICATION")
            return True
        return False

    def on_confirm_received(self) -> int:
        """Peer sent HANDOFF_CONFIRM with epoch e. Transition to CONFIRMED and release exclusivity."""
        if self.sender_state in (SenderState.IDENTITY_VERIFICATION, SenderState.ACQUISITION):
            self.sender_state = SenderState.CONFIRMED
            # Advance epoch to e+1 for ACK
            new_epoch = self.epoch_manager.advance_to_next_epoch()
            self._notify_change(f"VERIFICATION -> CONFIRMED (Epoch advanced to {new_epoch})")
            return new_epoch
        return self.epoch_manager.current_epoch

    def on_owner_claim_received(self) -> bool:
        """Peer published OWNER_CLAIM (e+1): demote to STANDBY."""
        if self.sender_state == SenderState.CONFIRMED:
            self.sender_state = SenderState.STANDBY
            self.role = OwnershipRole.STANDBY
            self.active_transaction_id = None
            self._notify_change("CONFIRMED -> STANDBY (Ownership transfer complete)")
            return True
        return False

    def abort_to_recovery(self, reason: str) -> int:
        """Watchdog timeout, candidate dead, or T_claim expired: abort and reclaim at e+2."""
        if self.sender_state not in (SenderState.STANDBY, SenderState.TRACKING):
            self.sender_state = SenderState.RECOVERY
            self.role = OwnershipRole.OWNER
            # Advance epoch to e+2 to preempt any stale e+1 claims
            reclaim_epoch = self.epoch_manager.advance_to_recovery_epoch()
            self._notify_change(f"ABORT to RECOVERY: {reason} (Reclaimed at epoch {reclaim_epoch})")
            return reclaim_epoch
        return self.epoch_manager.current_epoch

    def recovery_to_solo_tracking(self):
        """Resume solo tracking after aborting failed handoff."""
        if self.sender_state == SenderState.RECOVERY:
            self.sender_state = SenderState.TRACKING
            self.role = OwnershipRole.OWNER
            self.active_transaction_id = None
            self._notify_change("RECOVERY -> TRACKING (Solo tracking active)")

    # --- RECEIVER (ROBOT B) TRANSITIONS ---
    def on_request_received(self, txn_id: str, target_id: str) -> bool:
        """Incoming HANDOFF_REQUEST received."""
        if self.role == OwnershipRole.STANDBY and self.receiver_state == ReceiverState.IDLE:
            self.receiver_state = ReceiverState.EVALUATING
            self.active_transaction_id = txn_id
            self.active_target_id = target_id
            self._notify_change("IDLE -> EVALUATING")
            return True
        return False

    def accept_request(self) -> bool:
        """Passed hard gate G(j) and utility feasibility: accept."""
        if self.receiver_state == ReceiverState.EVALUATING:
            self.receiver_state = ReceiverState.ACCEPTED
            self.role = OwnershipRole.CANDIDATE
            self._notify_change("EVALUATING -> ACCEPTED (Role=CANDIDATE)")
            return True
        return False

    def start_acquisition(self) -> bool:
        """Begin slewing toward target spatial prior."""
        if self.receiver_state == ReceiverState.ACCEPTED:
            self.receiver_state = ReceiverState.SLEWING
            self._notify_change("ACCEPTED -> SLEWING")
            return True
        return False

    def target_locked(self) -> bool:
        """Target locked in FOV: proceed to Re-ID verification."""
        if self.receiver_state in (ReceiverState.SLEWING, ReceiverState.SEARCHING):
            self.receiver_state = ReceiverState.VERIFYING
            self._notify_change("SLEWING -> VERIFYING")
            return True
        return False

    def send_confirm(self) -> bool:
        """Re-ID verified >= 0.70: send HANDOFF_CONFIRM, wait for ACK (T_ack = 100ms)."""
        if self.receiver_state == ReceiverState.VERIFYING:
            self.receiver_state = ReceiverState.AWAITING_ACK
            self._notify_change("VERIFYING -> AWAITING_ACK (T_ack timer armed)")
            return True
        return False

    def on_confirm_ack_received(self, ack_epoch: int) -> bool:
        """Received HANDOFF_CONFIRM_ACK with epoch e+1: BECOME OWNER!"""
        if self.receiver_state == ReceiverState.AWAITING_ACK:
            self.epoch_manager.set_epoch(ack_epoch)
            self.receiver_state = ReceiverState.TRACKING
            self.role = OwnershipRole.OWNER
            self._notify_change(f"AWAITING_ACK -> TRACKING (Became OWNER at epoch {ack_epoch})")
            return True
        return False

    def on_ack_timeout(self):
        """T_ack (100ms) expired without ACK: cancel transaction and revert to IDLE."""
        if self.receiver_state == ReceiverState.AWAITING_ACK:
            self.receiver_state = ReceiverState.IDLE
            self.role = OwnershipRole.STANDBY
            self.active_transaction_id = None
            self._notify_change("AWAITING_ACK -> IDLE (T_ack timeout, transaction canceled)")

    def _notify_change(self, msg: str):
        if self.on_state_changed is not None:
            self.on_state_changed(self.robot_id, msg)
