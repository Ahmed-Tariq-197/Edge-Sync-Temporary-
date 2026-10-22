# EdgeSync Protocol: HandoffConfirm.msg (Math/Protocol Spec v2.3 §3.4)
edgesync_msgs/HandoffHeader header

float32     reid_similarity        # Visual similarity score (threshold >= 0.70)
string      frame_id               # Frame of reference ("map")
float32[3]  position               # Receiver-observed target 3D centroid
float32[21] cov_upper_triangle     # Receiver local track covariance upper triangle
