# EdgeSync Protocol: HandoffCancel.msg (Math/Protocol Spec v2.3 §4.2)
edgesync_msgs/HandoffHeader header

uint8 reason                      # Reason for canceling transaction

# Cancel Reason Constants
uint8 REASON_USER_ABORT           = 1
uint8 REASON_TRACK_LOST           = 2
uint8 REASON_ACK_TIMEOUT          = 3
uint8 REASON_WATCHDOG_CUTOFF      = 4
uint8 REASON_PEER_REJECTED        = 5
uint8 REASON_TARGET_OCCLUDED      = 6
uint8 REASON_EPOCH_MISMATCH       = 7
