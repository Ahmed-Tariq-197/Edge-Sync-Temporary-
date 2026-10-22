# EdgeSync Estimation: Target3DState.msg (C-02 -> C-03 / C-04 / System internal tracking)
string      target_id              # Target identifier
string      frame_id               # Spatial reference frame ("map")
float32[3]  position               # Position [x, y, z] in meters
float32[3]  velocity               # Velocity [vx, vy, vz] in m/s
float32[36] full_covariance        # Full 6x6 state covariance matrix (row-major)
float32     confidence             # Filter confidence / track quality score
int64       stamp_ns               # UTC timestamp in nanoseconds
