# EdgeSync Protocol: AcqStatus.msg (10 Hz Liveness Heartbeat, Math/Protocol Spec v2.3 §3.3)
edgesync_msgs/HandoffHeader header

uint8   status                    # Current receiver acquisition progress
float32 confidence                # Acquisition detection confidence [0.0 - 1.0]

# Status Constants
uint8 STATUS_SLEWING              = 1   # Rotating / pre-positioning chassis toward prior
uint8 STATUS_SEARCHING            = 2   # Searching target in FOV
uint8 STATUS_ACQUIRED             = 3   # Centroid detected and locked in FOV
