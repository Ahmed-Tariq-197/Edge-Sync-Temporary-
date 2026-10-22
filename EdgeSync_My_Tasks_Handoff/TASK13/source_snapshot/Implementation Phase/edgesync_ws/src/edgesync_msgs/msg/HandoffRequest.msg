# EdgeSync Protocol: HandoffRequest.msg (SRS FR-27, Math/Protocol Spec v2.3 §4.2)
edgesync_msgs/HandoffHeader header

int64   tto_deadline_utc_ns       # Absolute UTC nanoseconds at which target will be lost
int32   tto_remaining_ms_at_send  # Relative fallback margin (1230 - 2500 ms)
float32 tto_sigma_ms              # Uncertainty in TTO estimate

string  frame_id                  # Shared spatial reference frame ("map")
float32[3] position               # Target 3D position [x, y, z] in map frame (meters)
float32[3] velocity               # Target 3D velocity [vx, vy, vz] in map frame (m/s)
float32[21] cov_upper_triangle    # Target 6x6 state covariance upper triangle (symmetric)

float32[3] sender_pose            # Sender pose [x, y, yaw] in map frame
float32[6] sender_pose_cov_ut     # Sender 3x3 pose covariance upper triangle

float32[512] reid_feature_vector  # 512-dim L2-normalized OSNet visual embedding (2048 bytes)
float32 reid_quality              # Embedding confidence / image crop quality score [0.0 - 1.0]
