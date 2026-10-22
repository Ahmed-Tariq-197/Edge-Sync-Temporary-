# EdgeSync Prediction: TTOEstimate.msg (C-03 -> C-04 / Interface I-04)
string target_id
string frame_id
float32 tto_hat_s                    # Nominal analytical time-to-occlusion (s)
float32 sigma_tto_s                  # Projected uncertainty standard deviation (s)
float32 tto_safe_s                   # Uncertainty-bounded conservative margin: max(0, tto_hat - k_tau*sigma)
string exit_boundary                 # Boundary triggering visibility loss ("horizontal_right", "range_max", etc.)
geometry_msgs/Point[] predicted_trajectory # Discretized future 3D path in shared frame
int64 stamp_utc_ns                   # Observation timestamp in nanoseconds UTC
