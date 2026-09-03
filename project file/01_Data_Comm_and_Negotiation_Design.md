# Detailed Design Document
## Distributed Multi-Robot Tracking Handoff System
### Sections 1 & 2 — Data & Communication Protocol / Negotiation & Decision Logic

**Document status:** Detailed Design Phase — R&D Blueprint
**Owners:** Network & IoT Sub-team (Section 1) · AI & Algorithms Sub-team (Section 2)
**Related deliverables:** `negotiation_flow.mermaid`, `topic_taxonomy.mermaid`

---

## SECTION 1 — Data & Communication Protocol

### 1.1 MQTT Topic Taxonomy

A flat or ad-hoc topic space does not scale past a handful of robots — subscriptions become expensive, wildcard filtering gets ambiguous, and QoS cannot be tuned per data class. We therefore propose a **three-tier hierarchical namespace** that separates traffic by *function* first, then by *entity*, then by *stream*. This lets a broker-side ACL and each robot's subscription filter cleanly select "only telemetry," "only my track's bidding," etc., without parsing payloads.

```
fleet/
├── discovery/
│   └── heartbeat                          # presence/liveness beacons
│
├── <robot_id>/
│   ├── telemetry/
│   │   ├── pose                           # odometry / localization
│   │   ├── battery                        # SoC, voltage, health
│   │   └── status                         # mode, fault flags
│   └── vision/
│       └── tracks/<track_id>              # per-target Re-ID + kinematics
│
└── handoff/
    └── <track_id>/
        ├── announce                       # "I am losing this target"
        ├── bid                            # utility-score broadcast
        ├── award                          # winner declaration
        └── ack                            # winner confirms possession
```

**Design rationale**

| Namespace | Purpose | Publishers | Subscribers |
|---|---|---|---|
| `fleet/discovery/heartbeat` | Fleet membership / liveness | every robot | every robot |
| `fleet/<robot_id>/telemetry/*` | Per-robot health & pose | owning robot | fleet manager, neighbors within radio range |
| `fleet/<robot_id>/vision/tracks/<track_id>` | Raw Re-ID + kinematic observation | owning robot | robots subscribed to that specific `track_id` (dynamic subscribe on proximity) |
| `handoff/<track_id>/#` | Negotiation for a specific target | all candidate robots | all candidate robots |

Keying negotiation topics by `track_id` rather than `robot_id` means any robot can subscribe to *only the targets it is near*, using MQTT wildcards (`handoff/17/#`) instead of receiving fleet-wide chatter. This is the single biggest lever for reducing unnecessary bandwidth as fleet size grows, since negotiation traffic scales with **active occlusion events**, not with **fleet size²**.

**QoS assignment**

| Topic class | QoS | Retained | Justification |
|---|---|---|---|
| `telemetry/pose`, `telemetry/battery` | 0 | No | High frequency, latest value supersedes old — loss is acceptable |
| `vision/tracks/*` | 1 | No | Should arrive, occasional duplicate is harmless (consumer is idempotent on `ts`) |
| `handoff/*/announce`, `handoff/*/bid` | 1 | No | Must arrive for consensus to converge, duplicates tolerated (dedup by `seq`) |
| `handoff/*/award` | 2 | Yes (short TTL) | Exactly-once — a duplicate or dropped award causes two robots to claim the same target |
| `discovery/heartbeat` | 0 | Yes | Latest-state semantics; late joiners get current fleet snapshot on subscribe |

### 1.2 Lean JSON Payload Schema

The dominant cost in any handoff message is the **128-dimensional Re-ID embedding**. A naive encoding — JSON array of 128 float64 — costs roughly:

```
128 × ~18 chars/float (sign, digits, decimal, comma) ≈ 2,300 bytes
```

This alone would dwarf every other field combined and is unacceptable over a contended wireless edge link with many robots publishing concurrently. We reduce this in three stages:

**Stage 1 — Precision reduction (float64 → int8 quantization)**
Re-ID vectors from a trained embedding model are typically L2-normalized (‖v‖ = 1), so every component lies in a bounded, predictable range. We quantize each float to a signed 8-bit integer:

```
q_i = round(v_i × 127),   q_i ∈ [-127, 127]
```

Reconstruction on the receiving side: `v_i ≈ q_i / 127`. Empirically, int8 quantization of L2-normalized embeddings preserves cosine similarity ranking to well within the noise floor of Re-ID matching thresholds (matching decisions use similarity *ordering*, not exact values) — an acceptable, deliberate accuracy/bandwidth trade-off for a real-time handoff decision, as opposed to storage-tier re-identification.

This alone shrinks the vector from **1024 bytes (float64) / 512 bytes (float32) → 128 bytes**.

**Stage 2 — Binary-safe transport (avoid base64 in JSON)**
If the vector must live inside a JSON text payload, base64 adds ~33% overhead (128 bytes → 172 chars). We recommend **not** forcing the vector into JSON text at all — see §1.3 for the CBOR/binary alternative, which carries the 128 raw bytes natively.

**Stage 3 — Short, non-descriptive keys**
Field names are repeated in every message; verbose keys like `"reid_embedding_vector"` cost more than the data they label at this payload size. We use a fixed 2–3 letter key contract, documented once (below) rather than self-described per message.

**Proposed schema — `handoff/<track_id>/bid` (JSON representation)**

```json
{
  "tid": 17,
  "rid": "R04",
  "ts": 1893456213,
  "seq": 42,
  "pos": [12.4, 5.1],
  "vel": [0.8, -0.2],
  "conf": 0.91,
  "batt": 0.63,
  "los": 0.78,
  "u": 0.712,
  "reid": "b64:qk3F...=="
}
```

**Field dictionary**

| Key | Type | Bytes (typical) | Meaning |
|---|---|---|---|
| `tid` | uint | 2–3 | Track ID being negotiated |
| `rid` | string | 4 | Robot identifier (fixed-width code, not UUID) |
| `ts` | uint32 | 4 | Unix epoch, seconds (or ms if sub-second handoff timing is required) |
| `seq` | uint16 | 2 | Monotonic sequence per robot, for dedup/ordering |
| `pos` | [int16, int16] | 4 | Position, cm resolution, relative to map origin |
| `vel` | [int16, int16] | 4 | Velocity, cm/s |
| `conf` | uint8 (0–255) | 1 | Detection confidence, scaled from `[0,1]` |
| `batt` | uint8 (0–255) | 1 | Battery SoC, scaled from `[0,1]` |
| `los` | uint8 (0–255) | 1 | Line-of-sight probability, scaled from `[0,1]` |
| `u` | uint16 | 2 | Precomputed utility score ×10000, avoids re-derivation by receivers |
| `reid` | bytes[128] | 128 | Quantized Re-ID vector (see Stage 1) |

Total raw payload ≈ **150–155 bytes** in binary form (CBOR/MessagePack), versus **~2.6 KB** for the naive float64/verbose-key JSON version — roughly a **17× reduction**, almost entirely attributable to Re-ID quantization plus binary (non-base64) transport.

### 1.3 Serialization & Compression Strategy

| Option | Payload for schema above | Verdict |
|---|---|---|
| JSON (float64, verbose keys) | ~2,600 B | Baseline — rejected |
| JSON (int8 reid via base64, short keys) | ~330 B | Acceptable fallback where a JSON-only broker/tooling constraint exists |
| **CBOR** (binary, short keys, native byte-string for `reid`) | **~155 B** | **Recommended** — schema-flexible like JSON but binary-native, avoids base64 tax entirely, well-supported MQTT/ROS 2 bridge libraries exist |
| Protobuf (fixed schema, varint ints) | ~150 B | Marginally smaller than CBOR for scalar fields, but requires a compiled `.proto` contract on every robot — worth it only once the schema is stable/frozen |
| MessagePack | ~150 B | Comparable to CBOR; CBOR preferred for its IETF standardization and native ROS/MQTT tooling maturity |

**Recommendation:** adopt **CBOR** for all `handoff/*` and `vision/tracks/*` payloads now, with a migration path to **Protobuf** once the message contract stabilizes past the simulation phase (Protobuf gives compile-time schema safety, which matters more once multiple sub-teams integrate).

**On general-purpose compression (gzip/deflate/zlib):** we recommend **against** applying it to the Re-ID vector specifically — after int8 quantization the vector is near-maximum entropy (it is effectively noise-like data from a compressor's perspective), so deflate typically *adds* overhead rather than removing it. General compression remains useful only for **batched telemetry** (e.g., a 1 Hz aggregate of many robots' pose history sent to a fleet dashboard), not for the latency-critical single-target handoff path.

**Latency-oriented transport notes:**
- Use **MQTT 5** topic aliases on high-frequency topics (`telemetry/*`, `vision/tracks/*`) to replace the full topic string with a 2-byte alias after the first message — meaningful savings when topic strings are longer than the payload itself.
- Keep `handoff/*/award` payloads minimal (`tid`, `rid`, `ts` only) and QoS 2, since this is the one message where correctness outweighs size.
- Disable MQTT 5 user properties / response-topic / correlation-data on hot-path topics unless a specific feature (e.g., request/response bidding) requires them — each adds bytes to every packet.

---

## SECTION 2 — Negotiation & Decision Logic (`NegotiationSERVICE`)

### 2.1 Utility Score Cost Function

The utility score must be a **single scalar**, cheaply computable on-board, monotonically reflecting "how good a candidate is this robot to take over tracking," and comparable across robots without needing a shared reference frame beyond the map.

We define a normalized, weighted-sum utility:

```
U_i = w1 · f_dist(d_i) + w2 · f_los(p_i) + w3 · f_batt(b_i) + w4 · f_load(n_i)

subject to:  w1 + w2 + w3 + w4 = 1,   U_i ∈ [0, 1]
```

**Component functions**

1. **Distance term** — Euclidean distance from candidate robot to the target's *predicted* position at handoff time (not current position, to account for target motion during negotiation latency):

```
d_i = ‖ p_robot_i − p_target(t + Δt_negotiation) ‖₂

f_dist(d_i) = max(0, 1 − d_i / d_max)
```
`d_max` is a tunable sensor-range cutoff (e.g., camera's effective detection range); beyond it, a robot contributes zero regardless of other factors, which also acts as a natural candidate filter.

2. **Line-of-sight term** — probability of unobstructed view, derived from raycasting against the local ROS 2 occupancy grid (see Section 3 deliverable) or from a rolling detection-success rate:

```
f_los(p_i) = p_i,   p_i ∈ [0, 1]     (already a probability — used directly, no reshaping)
```

3. **Battery term** — deliberately **non-linear**. Linear scaling under-penalizes low-battery robots that are otherwise well-positioned; we want a soft cutoff, not a small deduction:

```
f_batt(b_i) = b_i^γ ,   γ > 1  (e.g., γ = 2)
```
This concentrates the penalty near depletion (a robot at 20% battery contributes `0.2² = 0.04`, not `0.2`) while barely discounting robots above ~70%. Optionally hard-gate: any robot below a reserve threshold (e.g., 15%) is excluded from bidding entirely rather than merely down-weighted, since a mid-handoff shutdown is worse than a slightly suboptimal handoff.

4. **Load term** — discourages "hot" robots already tracking near their concurrent-target capacity from taking on more:

```
f_load(n_i) = 1 − (n_i / n_max)
```
where `n_i` is the robot's current active track count and `n_max` its declared tracking capacity.

**Proposed weights**

| Weight | Value | Rationale |
|---|---|---|
| `w1` (distance) | **0.35** | Strongest single predictor of sustained tracking quality — closer robots re-acquire faster and keep the target in higher-resolution FOV |
| `w2` (line-of-sight) | **0.30** | The entire handoff exists *because* of occlusion risk, so LOS must weigh nearly as heavily as raw proximity — a close robot with no LOS is a bad candidate |
| `w3` (battery) | **0.20** | Operationally critical but secondary to tracking quality; the γ=2 shaping already lets it dominate near depletion regardless of nominal weight |
| `w4` (load) | **0.15** | A load-balancing nudge rather than a primary criterion — prevents starvation of idle robots without overriding a clearly superior candidate |

These weights should be treated as a **tunable starting point**, not fixed constants — we recommend exposing them as a runtime-loadable config (ROS 2 parameter or CBOR config topic) so Section 4's Gazebo testbed can run weight-sensitivity sweeps and empirically re-tune before field deployment.

### 2.2 Decentralized Tie-Breaking Mechanism

Two robots broadcasting numerically identical (or near-identical) utility scores is a real risk, especially with quantized `u` values (§1.2 encodes utility at 4 decimal digits of precision — collisions are plausible). Without a master node, the fleet needs a **deterministic, communication-free** rule every robot can evaluate identically and independently.

**Proposed protocol — three-layer resolution:**

1. **Epsilon-banding:** treat scores within `ε` (e.g., `ε = 0.01`) as tied, rather than requiring bit-exact equality — this avoids fragile ties caused purely by quantization noise while still resolving genuinely close bids.

2. **Deterministic secondary key:** on a tie, every robot computes the same tiebreak key from data already present in each bid — no extra round-trip required:

```
key_i = hash(track_id ⊕ robot_id ⊕ seq)     (e.g., truncated SHA-1 or CRC32)

winner = argmin_i  key_i
```
Using a **hash of the ID pair** rather than raw `robot_id` comparison (e.g., "lowest ID wins") avoids a systemic bias where the same low-numbered robot always wins ties and gets overloaded — the hash re-randomizes the ordering per `track_id`, so tie-wins are distributed fleet-wide over time.

3. **Jittered bid broadcast (collision avoidance, not just resolution):** rather than all candidates broadcasting bids the instant they compute them, each robot waits a small random backoff `t_i ~ U(0, T_jitter)` before publishing (analogous to CSMA/CA). This reduces simultaneous-publish collisions on constrained wireless channels and, as a side benefit, lets a robot cancel its own bid if it observes another robot's *strictly higher* bid arrive during its own backoff window — reducing redundant traffic.

**Convergence rule (no master required):**
Each robot that is a candidate for `track_id`:
- Subscribes to `handoff/<track_id>/bid` for a fixed window `T_negotiate` after the `announce` message.
- Maintains the best `(u, key)` tuple seen so far, including its own.
- If, when `T_negotiate` elapses, its own bid is the best `(u, key)` tuple observed, it **self-declares** by publishing to `handoff/<track_id>/award` (QoS 2).
- All other candidates, on seeing `award`, stop bidding and the new owner publishes `ack`.
- **Safety net:** if `T_negotiate` elapses with *no* `award` seen (e.g., due to packet loss), each robot independently re-derives the winner from its own local bid set and re-announces — self-healing without any central arbiter, at the cost of one extra round if packets were lost.

This gives the fleet a fully decentralized, symmetric decision procedure: identical inputs (the set of received bids) deterministically produce identical outputs (the same winner) on every robot, without requiring a leader election protocol or centralized coordinator.

---

### Next steps
- Section 1 schema and topic taxonomy should be reviewed against the ROS 2 ↔ MQTT bridge library chosen for Section 4's Gazebo testbed to confirm CBOR support.
- Section 2 weights (`w1..w4`) and `T_negotiate`/`T_jitter` timing constants are prime candidates for the sensitivity sweeps run in the Level-1 Gazebo evaluation.
- See accompanying `topic_taxonomy.mermaid` and `negotiation_flow.mermaid` for the visual diagrams referenced above.
