# EdgeSync Protocol: HandoffResponse.msg (Math/Protocol Spec v2.3 §4.2)
edgesync_msgs/HandoffHeader header

uint8   decision                  # Decision code (ACCEPT / REJECT / UNAVAILABLE)
uint8   reason                    # Detailed admission / gate rejection reason
int32   est_acquisition_ms        # Estimated slew & intercept latency (<= 700 ms)
float32 cpu_load                  # Receiver CPU load percentage [0.0 - 100.0]
float32 battery                   # Receiver battery percentage [0.0 - 100.0]

# Decision Constants
uint8 DECISION_ACCEPT             = 1
uint8 DECISION_REJECT             = 2
uint8 DECISION_UNAVAILABLE        = 3

# Reason Constants
uint8 REASON_OK                   = 0
uint8 REASON_BUSY_CPU             = 1
uint8 REASON_LOW_BATTERY          = 2
uint8 REASON_NAV_UNAVAILABLE      = 3
uint8 REASON_LINK_STALE           = 4
uint8 REASON_TIME_INSUFFICIENT    = 5
uint8 REASON_OUT_OF_REACH         = 6
