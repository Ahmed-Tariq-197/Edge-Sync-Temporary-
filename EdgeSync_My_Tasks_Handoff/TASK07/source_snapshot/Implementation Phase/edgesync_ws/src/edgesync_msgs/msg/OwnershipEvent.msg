# EdgeSync Protocol: OwnershipEvent.msg (Reliable state transition / OWNER_CLAIM burst)
edgesync_msgs/HandoffHeader header

uint8 role                         # Asserted operational role (OWNER, CANDIDATE, STANDBY)
uint8 burst_index                  # Burst transmission counter (1..3) spaced 20 ms apart

# Role Constants
uint8 ROLE_OWNER                   = 1
uint8 ROLE_CANDIDATE               = 2
uint8 ROLE_STANDBY                 = 3
