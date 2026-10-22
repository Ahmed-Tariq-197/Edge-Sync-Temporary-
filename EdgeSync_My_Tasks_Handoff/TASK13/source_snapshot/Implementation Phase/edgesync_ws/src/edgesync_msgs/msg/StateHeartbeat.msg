# EdgeSync Protocol: StateHeartbeat.msg (1 Hz Telemetry & Telemetry Gating, Math/Protocol Spec v2.3 §4.2)
edgesync_msgs/HandoffHeader header

uint8   role                       # Peer operational role (1=OWNER, 2=CANDIDATE, 3=STANDBY)
float32 cpu_load                   # Receiver CPU load percentage [0.0 - 100.0]
float32 battery_pct                # Battery percentage [0.0 - 100.0]
uint8   nav_status                 # Navigation stack readiness (1=AVAILABLE, 0=DEGRADED)

# Role Constants
uint8 ROLE_OWNER                   = 1
uint8 ROLE_CANDIDATE               = 2
uint8 ROLE_STANDBY                 = 3

# Nav Status Constants
uint8 NAV_STATUS_DEGRADED          = 0
uint8 NAV_STATUS_AVAILABLE         = 1
