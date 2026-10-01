# 🚀 EdgeSync: Master Implementation & Engineering Execution Plan (v1.0)
## Proactive TTO-Driven Decentralized Target Handoff for Dual JetAuto UGVs
**Document Status:** FROZEN & CERTIFIED BASELINE  
**Author / Systems Architect:** Principal Systems Architect & Senior Robotics Engineer  
**Aligned Standards:** SRS v1.3 · SAD v2.3 · Math & Protocol Spec v2.3 · Experimental Design v1.1  
**Target Hardware:** Hiwonder JetAuto UGV (Dual Setup) · NVIDIA Jetson Orin Nano (8GB Unified Memory) · Astra Pro Plus RGB-D · RPLiDAR  
**Target Software Stack:** Ubuntu 22.04 LTS · ROS 2 Humble Hawksbill (`rclpy` exclusively) · FastDDS 2.x · TensorRT 8.6 Python API · PyCUDA / ONNX  

---

## 📌 Executive Summary & Architectural Traceability

This Implementation Plan provides an end-to-end, mathematically rigorous, phased blueprint to translate the frozen system design of **EdgeSync** into verified, production-ready **Python-exclusive (`rclpy`)** ROS 2 packages and embedded edge services. 

The implementation strictly satisfies the **5 System Invariants**:
1. **Zero Dual-Ownership Invariant:** The state `COMMITTED` preserves Robot A as sole owner; ownership transfers strictly upon atomic transmission of `HANDOFF_CONFIRM_ACK` with monotonic epoch `e+1`.
2. **Deterministic Latency Budget:** Total nominal execution budget $\Delta t_{exec} \le 866\\,\\mathrm{ms}$, with an empirical safety margin $\Delta t_{safety} = 364\\,\\mathrm{ms}$, bounding the total allowable handoff window to $\Delta t_{budget}^{max} = 1230\\,\\mathrm{ms}$.
3. **Anti-Deadlock Liveness Invariant:** Continuous 10 Hz `ACQ_STATUS` heartbeats with a 300 ms watchdog cutoff to immediately abort failed candidates (`ERR_CANDIDATE_DEAD`) and revert to solo tracking.
4. **Spatial Reference Consistency:** All inter-robot target states, velocities, and covariances are resolved in the shared `map` coordinate frame via `tf2_ros` and metric depth, with ego-motion noise strictly propagated into measurement covariance ($R_{total}$).
5. **Edge Resource & Resident Inference Invariant:** The 512-dimensional OSNet feature extractor is pre-warmed and resident in unified memory, ensuring zero engine-load latency during the active handoff window.

---

## 📐 Strict Python, SOLID & Testing Directives

To guarantee modularity, maintainability, and academic rigor, the implementation enforces the following non-negotiable architectural mandates:

1. **Python Exclusivity (`rclpy`):**
    * All computational and lifecycle nodes are implemented strictly in Python 3.10 using `rclpy`.
    * Standard ROS 2 Python package structures (`setup.py`, `setup.cfg`, `package.xml`, package namespace directories with `__init__.py`) are used across all functional packages.
    * **Only** `edgesync_msgs` utilizes `CMakeLists.txt` for standard ROS 2 IDL message generation (`rosidl_default_generators`).
2. **SOLID & OOP Design Enforcement (DIP & OCP):**
    * All AI and perception models (YOLO Detector, ByteTrack MOT, OSNet Feature Extractor) MUST inherit from Python Abstract Base Classes (`abc.ABC`).
    * Nodes must depend strictly on abstractions (e.g., `BaseDetector`, `BaseIdentityExtractor`, `BaseStateEstimator`, `BaseTTOPredictor`), never concrete classes. Concrete TensorRT/PyTorch implementations are injected dynamically via factory patterns, enabling model swapping with zero modification to node logic.
3. **Test-Driven Development (TDD) with `pytest`:**
    * All isolated mathematical algorithms (3D Depth Deprojection, 6-DoF EKF, Ego-Motion Covariance Propagation, Analytical TTO Frustum/Raycast) must have dedicated `pytest` unit test suites.
    * Unit tests must achieve 100% pass and $>90\%$ code coverage on pure mathematical logic **before** integration into `rclpy` subscription/timer callbacks.
4. **Zero-Hardcoding Mandate:**
    * **Zero hardcoded magic numbers** (e.g., 866 ms, 1230 ms, 300 ms, 700 ms, 0.70 similarity threshold, utility weights $w_1, w_2, w_3$) are permitted in Python source files.
    * All parameters, timing thresholds, covariance priors, and topic names must be declared and dynamically injected via ROS 2 parameter YAML files managed by `config_param_node`.

---

## 🏗️ System Architecture & Package Structure

```
edgesync_ws/
├── src/
│   ├── edgesync_msgs/                    # ROS 2 Interfaces (C++ / CMakeLists.txt for rosidl)
│   │   ├── msg/
│   │   │   ├── HandoffHeader.msg
│   │   │   ├── HandoffRequest.msg
│   │   │   ├── HandoffResponse.msg
│   │   │   ├── HandoffCancel.msg
│   │   │   ├── AcqStatus.msg
│   │   │   ├── HandoffConfirm.msg
│   │   │   ├── HandoffConfirmAck.msg
│   │   │   ├── StateHeartbeat.msg
│   │   │   └── TargetState3D.msg
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   ├── edgesync_perception/              # Python Package (setup.py, rclpy)
│   │   ├── edgesync_perception/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces/               # SOLID: Abstract Base Classes (DIP/OCP)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_detector.py      # class BaseDetector(ABC)
│   │   │   │   └── base_tracker.py       # class BaseTracker(ABC)
│   │   │   ├── models/                   # Concrete Model Implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── yolo_trt_detector.py  # class YoloTrtDetector(BaseDetector)
│   │   │   │   └── bytetrack_mot.py      # class ByteTrackMOT(BaseTracker)
│   │   │   ├── math/
│   │   │   │   ├── __init__.py
│   │   │   │   └── depth_deprojector.py  # Pure mathematical deprojection logic
│   │   │   └── nodes/
│   │   │       ├── __init__.py
│   │   │       └── perception_node.py    # rclpy.node.Node (Depends on BaseDetector)
│   │   ├── tests/                        # TDD: pytest suite
│   │   │   ├── __init__.py
│   │   │   └── test_depth_deprojection.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
│   │
│   ├── edgesync_estimation/              # Python Package (setup.py, rclpy)
│   │   ├── edgesync_estimation/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces/               # SOLID: Abstract Base Classes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_estimator.py     # class BaseStateEstimator(ABC)
│   │   │   │   └── base_tto_predictor.py # class BaseTTOPredictor(ABC)
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── target_ekf_3d.py      # 6-DoF EKF + Measurement Ego Covariance
│   │   │   │   ├── ego_covariance.py     # Jacobians J_c, J_xi & propagation
│   │   │   │   └── tto_calculator.py     # 3D Frustum exit & obstacle raycast
│   │   │   └── nodes/
│   │   │       ├── __init__.py
│   │   │       └── estimation_node.py    # rclpy.node.Node (Depends on abstractions)
│   │   ├── tests/                        # TDD: pytest suite
│   │   │   ├── __init__.py
│   │   │   ├── test_target_ekf.py
│   │   │   ├── test_ego_covariance.py
│   │   │   └── test_tto_calculator.py
│   │   ├── config/
│   │   │   └── estimation_params.yaml    # Zero-hardcoding injection
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
│   │
│   ├── edgesync_identity/                # Python Package (setup.py, rclpy)
│   │   ├── edgesync_identity/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces/               # SOLID: Abstract Base Class
│   │   │   │   ├── __init__.py
│   │   │   │   └── base_identity_extractor.py # class BaseIdentityExtractor(ABC)
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   └── osnet_trt_extractor.py     # class OsnetTrtExtractor(BaseIdentityExtractor)
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── identity_verifier.py       # Cosine similarity & multi-cue fusion
│   │   │   │   └── signature_smoother.py      # Exponential Moving Average buffer
│   │   │   └── nodes/
│   │   │       ├── __init__.py
│   │   │       └── identity_node.py           # rclpy.node.Node (Resident engine)
│   │   ├── tests/                        # TDD: pytest suite
│   │   │   ├── __init__.py
│   │   │   └── test_identity_fusion.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
│   │
│   ├── edgesync_coordination/            # Python Package (setup.py, rclpy)
│   │   ├── edgesync_coordination/
│   │   │   ├── __init__.py
│   │   │   ├── fsm/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ownership_fsm.py           # Explicit states & transitions
│   │   │   │   └── epoch_manager.py           # Monotonic epoch tracking & validation
│   │   │   ├── decision/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── eligibility_gate.py        # Hard Gate G(j)
│   │   │   │   └── utility_engine.py          # Utility U_j calculation
│   │   │   ├── protocol/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── negotiation_manager.py     # P2P message handler
│   │   │   │   └── liveness_watchdog.py       # 10Hz heartbeat timer & 300ms cutoff
│   │   │   └── nodes/
│   │   │       ├── __init__.py
│   │   │       └── coordination_node.py       # rclpy.node.Node
│   │   ├── tests/                        # TDD: pytest suite
│   │   │   ├── __init__.py
│   │   │   ├── test_fsm_transitions.py
│   │   │   ├── test_eligibility_gate.py
│   │   │   └── test_watchdog_timeout.py
│   │   ├── config/
│   │   │   └── coordination_params.yaml       # Zero-hardcoding injection
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
│   │
│   ├── edgesync_navigation/              # Python Package (setup.py, rclpy)
│   │   ├── edgesync_navigation/
│   │   │   ├── __init__.py
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── acquisition_controller.py  # Slew trajectory computation
│   │   │   │   └── safety_supervisor.py       # Safe-Stop & velocity ramp-down
│   │   │   └── nodes/
│   │   │       ├── __init__.py
│   │   │       ├── navigation_bridge_node.py  # Nav2 client integration
│   │   │       └── safety_authority_node.py   # Final motion authority (/cmd_vel mux)
│   │   ├── tests/                        # TDD: pytest suite
│   │   │   ├── __init__.py
│   │   │   └── test_safety_stop.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
│   │
│   ├── edgesync_observability/           # Python Package (setup.py, rclpy)
│   │   ├── edgesync_observability/
│   │   │   ├── __init__.py
│   │   │   ├── logging/
│   │   │   │   ├── __init__.py
│   │   │   │   └── event_logger.py            # Structured JSON/CSV timestamped logger
│   │   │   ├── evaluation/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ground_truth_loader.py     # Offline Mocap/Rig alignment
│   │   │   │   └── metrics_calculator.py      # HSR, ITA, TTO Error, Zero Dual-Owner
│   │   │   └── nodes/
│   │   │       ├── __init__.py
│   │   │       ├── monitoring_node.py         # Telemetry & tegrastats monitor
│   │   │       └── config_param_node.py       # Dynamic parameter server
│   │   ├── scripts/
│   │   │   └── run_offline_evaluation.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── setup.cfg
│   │
│   └── edgesync_bringup/                 # Launch & Configuration Package
│       ├── config/
│       │   ├── fastdds_profiles.xml
│       │   ├── hardware_params.yaml       # Injected constants
│       │   ├── timing_budgets.yaml        # 866ms, 1230ms, 300ms, 700ms injected here
│       │   └── algorithm_thresholds.yaml  # 0.70 similarity, 2.50s/1.23s zones
│       ├── launch/
│       │   ├── robot_bringup.launch.py
│       │   ├── simulation_tier1.launch.py
│       │   └── physical_tier3.launch.py
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
```

---

## 🗓️ Phased Implementation Roadmap (Python & SOLID Constrained)

```mermaid
gantt
    title EdgeSync Master Engineering Timeline (Python & TDD)
    dateFormat  YYYY-MM-DD
    section Phase 0: Foundations
    Sprint 1 - Edge Setup & FastDDS XML Config      :p0, 2026-10-05, 10d
    section Phase 1: Core Python Stack
    Sprint 2 - Messages & Custom Interfaces         :p1, 2026-10-15, 7d
    Sprint 3 - 3D Deprojection & TDD Perception     :p2, 2026-10-22, 12d
    Sprint 4 - 3D EKF & Safe TTO Predictor (TDD)   :p3, 2026-11-03, 12d
    Sprint 5 - Resident TensorRT OSNet (SOLID ABC)  :p4, 2026-11-15, 10d
    section Phase 2: Distributed Protocol
    Sprint 6 - Ownership FSM & 10Hz Watchdog        :p5, 2026-11-25, 12d
    Sprint 7 - Nav2 Slew & Safe-Stop Authority      :p6, 2026-12-07, 10d
    section Phase 3: Validation & Benchmarking
    Sprint 8 - Tier 1 Simulation & Fault Injection  :p7, 2026-12-17, 12d
    Sprint 9 - Tier 2 Orin Nano Calibration         :p8, 2026-12-29, 8d
    Sprint 10 - Tier 3 Physical Testbed Validation  :p9, 2027-01-06, 14d
```

---

### Detailed Sprint Work Breakdown Structure (WBS)

#### 🔹 Phase 0: Workspace Provisioning & Communication Backbone
##### **Sprint 1: Jetson Orin Nano Edge Environment & DDS Configuration (Days 1–10)**
* **Deliverables:**
  1. Setup Ubuntu 22.04 LTS (JetPack 5.1.2 / 6.0) with Python 3.10 and ROS 2 Humble.
  2. Configure power profiles: `sudo nvpmodel -m 0` (15 W 6-core mode) and lock clocks via `sudo jetson_clocks`.
  3. Deploy **PTP (IEEE 1588v2)** time daemon (`ptp4l` / `chrony`) on 5 GHz Wi-Fi. Verify offset $\delta_{sync} \le 2\\,\\mathrm{ms}$.
  4. Author `fastdds_profiles.xml` for zero-copy intra-process transport and strict QoS (Reliable protocol msgs, Best-Effort 10 Hz Heartbeat with 150 ms deadline and 300 ms lease).
* **Quality Gate 1:** Ping RTT $\le 1.5\\,\\mathrm{ms}$; clock offset $\delta_{sync} \le 2.0\\,\\mathrm{ms}$ verified across 10,000 samples.

---

#### 🔹 Phase 1: Data Contracts, Local Perception & State Estimation (TDD Mandate)
##### **Sprint 2: Message Definitions & Serialization Layer (Days 11–17)**
* **Deliverables:**
  1. Implement package `edgesync_msgs` (using standard `CMakeLists.txt` for `rosidl_default_generators`).
  2. Implement `HandoffHeader.msg` enforcing protocol version, sender/receiver UUIDs, transaction ID, target ID, monotonic ownership epoch $e$, sequence numbers, and nanosecond UTC timestamps.
  3. Formulate `HandoffRequest.msg` with static array `float32[512] reid_feature_vector` and 21-element upper-triangular covariance (`float32[21] cov_upper_triangle`).
* **Quality Gate 2:** Validate total binary CDR payload $\le 3072\text{ bytes}$ via `ros2 topic echo` binary inspection.

##### **Sprint 3: 2D-to-3D Perception Engine & TDD Unit Testing (C-01) (Days 18–29)**
* **SOLID & OOP Requirements:**
   * Define `BaseDetector(abc.ABC)` with abstract method `detect(image: np.ndarray) -> List[Detection]`.
   * Define `BaseTracker(abc.ABC)` with abstract method `update(detections: List[Detection]) -> List[Track]`.
   * Implement concrete `YoloTrtDetector(BaseDetector)` using TensorRT Python API.
   * Node `perception_node.py` receives `BaseDetector` via dependency injection.
* **TDD Unit Testing Milestone (Mandatory Before ROS 2 Integration):**
   * Implement pure mathematical module `edgesync_perception.math.depth_deprojector`.
   * Write `pytest` test suite `test_depth_deprojection.py`:
     * Test median depth extraction across central 50% ROI against synthetic depth maps with noisy background boundaries.
     * Test pinhole deprojection formula against known ground-truth 3D points.
     * Test handling of `NaN`, zero depth, and out-of-range sensor readings.
* **Deliverables:**
  1. Complete and pass `pytest tests/test_depth_deprojection.py` (100% pass).
  2. Implement `perception_node.py` utilizing `tf2_ros` Buffer and TransformListener to transform deprojected camera coordinates to the shared `map` frame at image capture timestamp $t_{stamp}$.
* **Quality Gate 3:** Pure math unit tests pass 100%; perception node sustains $\ge 20\text{ FPS}$ on Orin Nano with 3D centroid Euclidean error $\le 0.15\text{ m}$.

##### **Sprint 4: 3D EKF Tracking, Ego-Motion Covariance & Safe TTO Predictor (C-02 & C-03) (Days 30–41)**
* **SOLID & OOP Requirements:**
   * Define `BaseStateEstimator(abc.ABC)` with abstract methods `predict(dt)` and `update(measurement, R_total)`.
   * Define `BaseTTOPredictor(abc.ABC)` with abstract method `compute_tto(target_state, camera_pose, obstacles) -> TTOResult`.
   * Implement concrete `TargetEKF3D(BaseStateEstimator)` and `AnalyticalTTOCalculator(BaseTTOPredictor)`.
   * Node `estimation_node.py` depends solely on abstractions.
* **TDD Unit Testing Milestone (Mandatory Before ROS 2 Integration):**
   * Write `pytest tests/test_target_ekf.py`:
     * Verify state transition matrix $F$ and constant velocity propagation.
     * Test Kalman gain and covariance convergence under synthetic noisy 3D observations.
   * Write `pytest tests/test_ego_covariance.py`:
     * Validate Jacobians $J_c = \frac{\partial p_m}{\partial p_c}$ and $J_\xi = \frac{\partial p_m}{\partial \xi}$.
     * Assert measurement covariance correctly computes $R_{total} = J_c R_{cam} J_c^T + J_\xi \Sigma_\xi J_\xi^T$ without process-noise double counting.
   * Write `pytest tests/test_tto_calculator.py`:
     * Test 3D FOV frustum intersection boundary conditions.
     * Test obstacle polygon raycasting and verify $T_{TTO}^{safe} = \max(0, \hat{T}_{TTO} - 2\sigma_{TTO})$.
* **Zero-Hardcoding Rule:** All EKF noise priors ($Q_{target}, R_{cam}$), FOV angles ($\alpha_{fov}, \beta_{fov}$), and zone boundaries (2.50 s, 1.23 s) injected dynamically from `estimation_params.yaml`.
* **Deliverables:**
  1. Complete and pass all 3 `pytest` test suites.
  2. Integrate `estimation_node.py` publishing `/target/state_3d` and `/target/tto`.
* **Quality Gate 4:** 100% pass on all estimation unit tests; EKF covariance remains positive semi-definite; $T_{TTO}^{safe}$ achieves $\ge 95\%$ conservative lower-bound coverage.

##### **Sprint 5: Resident TensorRT Identity Engine (C-07) (Days 42–51)**
* **SOLID & OOP Requirements:**
   * Define `BaseIdentityExtractor(abc.ABC)` with abstract method `extract_embedding(crop: np.ndarray) -> np.ndarray`.
   * Implement concrete `OsnetTrtExtractor(BaseIdentityExtractor)` using TensorRT Python runtime.
   * Node `identity_node.py` depends on `BaseIdentityExtractor`.
* **Deliverables:**
  1. Ensure OSNet engine is loaded during node `on_configure` / initialization and kept **resident in unified memory**.
  2. Implement `signature_smoother.py`: update exponential moving average buffer at 2 Hz ($\mathbf{f}_{target}(t) = \alpha \mathbf{f}_{new} + (1-\alpha) \mathbf{f}_{target}(t-1)$).
  3. Implement `identity_verifier.py`: Compute L2-normalized cosine similarity with threshold dynamically injected from `algorithm_thresholds.yaml` (default: 0.70).
  4. Write `pytest tests/test_identity_fusion.py` testing cosine distance and multi-cue rejection of distractors.
* **Quality Gate 5:** TensorRT OSNet inference latency on Orin Nano $\le 18\\,\\mathrm{ms}$; cosine similarity calculation $\le 0.5\\,\\mathrm{ms}$; zero dynamic allocations during handoff.

---

#### 🔹 Phase 2: Distributed Protocol, State Machine & Navigation Authority
##### **Sprint 6: Ownership FSM, P2P Coordination & 10 Hz Watchdog (C-04, C-05, C-06, C-09) (Days 52–63)**
* **Deliverables:**
  1. Implement `ownership_fsm.py` conforming strictly to the certified State Machine and Sequence Diagrams.
  2. Implement Robot A Sender FSM: `TRACKING` $\to$ `CANDIDATE` $\to$ `NEGOTIATING` $\to$ `COMMITTED` $\to$ `AWAITING_CONFIRMATION` $\to$ `CONFIRMED` / `RECOVERY`.
  3. Implement Robot B Receiver FSM: `IDLE` $\to$ `EVALUATING` $\to$ `ACCEPTED` $\to$ `SLEWING` $\to$ `SEARCHING` $\to$ `VERIFYING_IDENTITY` $\to$ `AWAITING_ACK` $\to$ `TRACKING_OWNER`.
  4. Implement `eligibility_gate.py` (Hard gate $G(j)$) and `utility_engine.py` ($U_j$) with weights ($w_1=0.5, w_2=0.3, w_3=0.2$) dynamically loaded from `coordination_params.yaml`.
  5. Implement `liveness_watchdog.py`:
      * Robot B publishes `AcqStatus.msg` at 10 Hz.
      * Robot A arms timer $\tau_{hb} = 300\\,\\mathrm{ms}$ (injected from YAML). Upon 3 missed heartbeats $\to$ raise `ERR_CANDIDATE_DEAD`, transition to `RECOVERY`, retain sole ownership.
  6. Implement **Strict Epoch Handshake**:
      * Robot B sends `HANDOFF_CONFIRM` with epoch $e$.
      * Robot A replies `HANDOFF_CONFIRM_ACK` with epoch $e+1$.
      * Robot A disarms watchdog, transitions to `PASSIVE_TRACKER`, and **only then** releases exclusive ownership.
      * Robot B becomes exclusive `TRACKING_OWNER` upon receiving the ACK.
  7. Write `pytest tests/test_fsm_transitions.py` and `test_watchdog_timeout.py`.
* **Quality Gate 6:** Zero dual-ownership violations across 500 simulated negotiation cycles; watchdog triggers within $320\\,\\mathrm{ms}$ upon simulated receiver SIGKILL.

##### **Sprint 7: Acquisition Slew, Nav2 Integration & Chassis Safety Authority (C-08 & C-10) (Days 64–73)**
* **Deliverables:**
  1. Implement `navigation_bridge_node.py`: wrap Nav2 Simple Commander Python API (`BasicNavigator`) for Mecanum base navigation.
  2. Implement `acquisition_controller.py`: compute pre-positioning intercept pose in shared `map` frame using extrapolated 3D velocity.
  3. Implement `safety_authority_node.py` (C-10):
      * C-10 is the ultimate `/cmd_vel` authority. Motion requests from C-08 are advisory.
      * Implement autonomous **Safe-Stop**: if localization is lost or communication severed, ramp velocity down to 0 at $1.5\text{ m/s}^2$.
* **Quality Gate 7:** Chassis slews to intercept orientation within $\le 450\\,\\mathrm{ms}$; emergency safe-stop engages within $50\\,\\mathrm{ms}$ of heartbeat severance.

---

#### 🔹 Phase 3: Multi-Tier Verification, Benchmarking & Physical Defense
##### **Sprint 8: Tier 1 Simulation & Fault Injection Harness (Days 74–85)**
* **Deliverables:**
  1. Setup Gazebo / ROS 2 simulation with 2 JetAuto UGV models and 1 moving human target actor.
  2. Implement automated Python test harness (`test_fault_injection.py`) executing Scenarios D (Crash), F (Netem loss/delay), and G (Stale epoch $e-1$).
* **Quality Gate 8:** 100% pass on protocol conformance suite; zero dual-ownership occurrences.

##### **Sprint 9: Tier 2 Edge Profiling & Constant Calibration (Days 86–93)**
* **Deliverables:**
  1. Deploy Python stack to physical NVIDIA Jetson Orin Nano 8GB hardware.
  2. Run 10-minute thermal soak running full perception, estimation, and ROS 2 graph.
  3. Execute automated profiling script measuring $\Delta t_{dec}, \Delta t_{neg}, \Delta t_{acq}, \Delta t_{id}, \Delta t_{conf}$, clock offset $\delta_{sync}$, and execution variance $\sigma_{total}$.
  4. Compute calibrated safety margin:
     $$\Delta t_{safety}^{calibrated} = 3 \sigma_{total}^{measured} + 2 \delta_{sync}^{measured}$$
  5. Freeze calibrated values into production `timing_budgets.yaml`.
* **Quality Gate 9:** Sustained perception rate $\ge 15\text{ FPS}$ under 100% stack load; total measured execution $\Delta t_{exec}^{p99} \le 866\\,\\mathrm{ms}$; zero thermal throttling.

##### **Sprint 10: Tier 3 Physical Testbed Validation & Ground Truth (Days 94–107)**
* **Deliverables:**
  1. Deploy dual physical JetAuto UGVs in indoor test arena equipped with external ground-truth motion tracking.
  2. Execute all 7 pre-registered experimental scenarios (Scenarios A through G).
  3. Run offline evaluation pipeline (`run_offline_evaluation.py`):
      * Calculate Handoff Success Rate ($\text{HSR} \ge 85\%$).
      * Calculate Identity Transfer Accuracy ($\text{ITA} \ge 90\%$).
      * Verify Dual-Ownership Violations $\equiv 0$.
      * Verify Zero Loss of Target Identity.
* **Quality Gate 10:** Complete statistical validation report with confidence intervals meeting all pre-registered success criteria in SRS v1.3 and Experimental Design v1.1.
