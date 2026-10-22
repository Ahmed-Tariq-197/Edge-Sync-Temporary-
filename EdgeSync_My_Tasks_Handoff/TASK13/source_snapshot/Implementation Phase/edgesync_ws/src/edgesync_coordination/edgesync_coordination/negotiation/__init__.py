"""Task-allocation negotiators (Experiment 13).

Importable without ROS: nothing in this subpackage imports ``rclpy`` or ``edgesync_msgs``, so the
allocation logic is unit-testable off a ROS graph.
"""
from .cbba_negotiator import (
    ALGORITHM_VARIANT,
    BELIEF_SCHEMA_VERSION,
    MIN_AGREEMENT_ROUND,
    PROTOCOL_ID,
    CBBANegotiator,
    ConsensusMessage,
    ConsensusStats,
    ConsensusStatus,
    RejectionClass,
    TaskInfo,
)

__all__ = [
    'ALGORITHM_VARIANT',
    'BELIEF_SCHEMA_VERSION',
    'MIN_AGREEMENT_ROUND',
    'PROTOCOL_ID',
    'CBBANegotiator',
    'ConsensusMessage',
    'ConsensusStats',
    'ConsensusStatus',
    'RejectionClass',
    'TaskInfo',
]
