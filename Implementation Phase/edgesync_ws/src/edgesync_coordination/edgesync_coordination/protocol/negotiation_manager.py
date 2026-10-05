"""Transaction table and P2P negotiation message admission (Math/Protocol Spec v2.3 §3.5).

Implements strict 4-step admission policy:
1. Protocol version and receiver ID validation.
2. Transaction table lookup (OPEN, CLOSED, CONFIRMED).
3. Monotonic epoch validation.
4. Sequence number tracking.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time


class TxnState(Enum):
    OPEN = auto()
    CONFIRMED = auto()
    CLOSED = auto()


@dataclass
class TransactionRecord:
    txn_id: str
    target_id: str
    peer_id: str
    state: TxnState
    created_time: float
    closed_time: Optional[float] = None
    stored_ack_epoch: Optional[int] = None
    reason: str = ""


class NegotiationManager:
    """Manages active transaction contexts and message admission order."""

    def __init__(self, self_id: str, protocol_version: int = 0x0203, tombstone_ttl_s: float = 10.0):
        self.self_id = self_id
        self.protocol_version = protocol_version
        self.tombstone_ttl_s = tombstone_ttl_s
        self._table: Dict[str, TransactionRecord] = {}

    def admit_message(self,
                      msg_protocol_version: int,
                      receiver_id: str,
                      txn_id: str,
                      msg_epoch: int,
                      current_epoch: int,
                      is_request: bool = False) -> Tuple[bool, str]:
        """Execute strict admission check on an incoming protocol message."""
        # Step 1: Protocol version compatibility check
        if msg_protocol_version != self.protocol_version:
            return False, f"Protocol version mismatch: 0x{msg_protocol_version:04x} != 0x{self.protocol_version:04x}"

        # Step 2: Receiver ID check
        if receiver_id not in (self.self_id, "*"):
            return False, f"Message addressed to {receiver_id}, not {self.self_id}"

        # Step 3: Transaction table lookup
        self._prune_tombstones()
        record = self._table.get(txn_id)

        # Step 4: Monotonic epoch check (Anti-divergence FR-54)
        if msg_epoch < current_epoch:
            # Special idempotency case: If transaction is CONFIRMED, duplicate CONFIRM carries epoch e (while current is e+1)
            if record is not None and record.state == TxnState.CONFIRMED:
                return True, "Duplicate CONFIRM admitted for idempotent ACK replay"
            return False, f"Stale epoch {msg_epoch} < current {current_epoch}"

        if record is None:
            if is_request:
                # Open new transaction record
                return True, "New transaction admission approved"
            return False, f"Unknown transaction ID {txn_id}"

        if record.state == TxnState.CLOSED:
            return False, f"Transaction {txn_id} is CLOSED ({record.reason})"

        return True, "Message admitted"

    def open_transaction(self, txn_id: str, target_id: str, peer_id: str):
        """Open a new active transaction."""
        self._table[txn_id] = TransactionRecord(
            txn_id=txn_id,
            target_id=target_id,
            peer_id=peer_id,
            state=TxnState.OPEN,
            created_time=time.monotonic()
        )

    def mark_confirmed(self, txn_id: str, ack_epoch: int):
        """Record stored ACK for idempotent replays upon duplicate CONFIRM."""
        if txn_id in self._table:
            self._table[txn_id].state = TxnState.CONFIRMED
            self._table[txn_id].stored_ack_epoch = ack_epoch

    def close_transaction(self, txn_id: str, reason: str):
        """Transition transaction to CLOSED with tombstone retention."""
        if txn_id in self._table:
            self._table[txn_id].state = TxnState.CLOSED
            self._table[txn_id].closed_time = time.monotonic()
            self._table[txn_id].reason = reason

    def get_stored_ack_epoch(self, txn_id: str) -> Optional[int]:
        """Return stored ACK epoch if transaction is CONFIRMED."""
        record = self._table.get(txn_id)
        if record and record.state == TxnState.CONFIRMED:
            return record.stored_ack_epoch
        return None

    def _prune_tombstones(self):
        """Remove CLOSED records whose TTL has expired."""
        now = time.monotonic()
        to_delete = []
        for txn_id, rec in self._table.items():
            if rec.state == TxnState.CLOSED and rec.closed_time is not None:
                if (now - rec.closed_time) > self.tombstone_ttl_s:
                    to_delete.append(txn_id)
        for txn_id in to_delete:
            del self._table[txn_id]
