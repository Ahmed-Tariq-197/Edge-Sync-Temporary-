# EdgeSync Protocol: Common Handoff Header (SAD v2.3 & Math/Protocol Spec v2.3)
uint16 protocol_version      # 0x0203 (v2.3)
uint8  message_type          # Message type discriminator
string sender_id             # Sender UGV UUID / Identifier (e.g., "ugv_1")
string receiver_id           # Receiver UGV UUID or "*" for broadcast
string transaction_id        # Unique transaction UUID
string target_id             # Physical target identifier
uint64 ownership_epoch       # Monotonic ownership counter (Anti-divergence FR-54)
uint32 sequence              # Per-transaction sequence number
int64  stamp_utc_ns          # PTP IEEE 1588v2 UTC timestamp in nanoseconds

# Message Type Constants
uint8 MSG_TYPE_REQUEST          = 1
uint8 MSG_TYPE_RESPONSE         = 2
uint8 MSG_TYPE_CANCEL           = 3
uint8 MSG_TYPE_ACQ_STATUS       = 4
uint8 MSG_TYPE_CONFIRM          = 5
uint8 MSG_TYPE_CONFIRM_ACK      = 6
uint8 MSG_TYPE_OWNERSHIP_EVENT  = 7
uint8 MSG_TYPE_STATE_HEARTBEAT  = 8
