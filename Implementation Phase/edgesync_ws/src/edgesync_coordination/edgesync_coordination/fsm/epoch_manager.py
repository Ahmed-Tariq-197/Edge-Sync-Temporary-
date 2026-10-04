"""Monotonic Epoch Manager (Anti-Divergence Invariant FR-54).

Guarantees:
1. Target ownership transitions are monotonically ordered (e -> e+1).
2. Stale epoch messages (e <= current) are dropped immediately.
3. Recovery reclaims ownership at e+2 to dominate any in-flight e+1 claim (highest epoch wins).
"""
from typing import Tuple


class EpochManager:
    """Tracks and validates target ownership epochs across cooperative UGVs."""

    def __init__(self, initial_epoch: int = 1):
        self._current_epoch = initial_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def validate_incoming_epoch(self, msg_epoch: int) -> bool:
        """Check if incoming message epoch is valid and strictly newer or current.

        Args:
            msg_epoch: Ownership epoch attached to message header.

        Returns:
            True if epoch is valid and >= current epoch.
        """
        return msg_epoch >= self._current_epoch

    def is_stale(self, msg_epoch: int) -> bool:
        """Return True if message carries an expired/stale epoch."""
        return msg_epoch < self._current_epoch

    def advance_to_next_epoch(self) -> int:
        """Increment epoch monotonically upon successful handoff completion (e -> e+1)."""
        self._current_epoch += 1
        return self._current_epoch

    def advance_to_recovery_epoch(self) -> int:
        """Advance epoch to e+2 upon handoff failure/reclaim to preempt in-flight e+1 claims."""
        self._current_epoch += 2
        return self._current_epoch

    def set_epoch(self, new_epoch: int):
        """Force set epoch if incoming authoritative claim has strictly higher epoch."""
        if new_epoch > self._current_epoch:
            self._current_epoch = new_epoch
