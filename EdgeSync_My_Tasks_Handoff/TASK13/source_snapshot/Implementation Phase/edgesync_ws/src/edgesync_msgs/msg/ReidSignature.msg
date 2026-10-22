# EdgeSync Experiments 7 <-> 13 contract: ReidSignature.msg
#
# Produced by edgesync_identity (Exp-7) for the robot that currently observes the target, consumed by
# edgesync_coordination when it fills in HandoffRequest.reid_feature_vector. It exists because the
# reference embedding had no carrier: IdentityNode could compute a smoothed signature but published
# nothing, and CoordinationNode.set_reid_signature() had no caller, so every HandoffRequest left with
# the zeros it was initialised with.
#
# Additive type. No existing message, service or field was changed.

string sender_id             # Robot id that produced this signature; a consumer must not adopt a peer's
string target_id             # Tracked target this embedding describes ('' when the producer does not know)
bool     valid               # False is an explicit invalidation, never a zero vector pretending to be one
float32[512] vector          # L2-normalized 512-D embedding; only meaningful while valid is true
float32 quality              # NOT an identity-match confidence and NOT a detector score. 0.0 means
                             # "the producer measured nothing": no Exp-7 path measures a quality
                             # today, so a consumer must read 0.0 as unavailable, never as a low
                             # score. A future producer that has a defensible measurement must state
                             # what it measured before anything consumes it.
uint64 stamp_utc_ns          # Observation time, in nanoseconds, in the PRODUCER's clock domain. It is
                             # Unix UTC only when that node runs the default system-time clock and not
                             # use_sim_time. Ordering two publications from the same producer is safe
                             # (one clock); comparing this against a *different* host's clock is not,
                             # so the consumer records its own arrival time too and refuses a stamp
                             # implausibly ahead of its clock instead of inferring freshness.
string model_id              # Extractor that produced it (e.g. transreid), so a mock can be identified
