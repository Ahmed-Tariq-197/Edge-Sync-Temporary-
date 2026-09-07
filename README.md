# Phase 2 Work Breakdown: Data Flow, Hardware/Software, and Interfaces

### Task 2.1: End-to-End Data Flow Architecture Specification
* **Objective:** Define data lifecycles across intra-UGV pipelines, inter-UGV metadata transfers, and receiver-side acquisition workflows.
* **Dependencies:** 
  * SAD: Section 1.1, Section 1.3, Section 3.3, Section 3.4, Section 3.5.
  * SRS: FR-01–FR-06, FR-14–FR-17, FR-26[cite: 5].
* **Deliverables:**
  * DF-01: Intra-UGV data pipeline specification (C-01 -> C-02 -> C-03 -> C-04 -> C-09).
  * DF-02: Inter-UGV metadata exchange flow (C-06_A <-> C-13 <-> C-06_B).
  * DF-03: Receiver-side acquisition and verification sequence (C-08 <-> C-07 -> C-09).
  * DF-04: Data flow matrix (Source, Destination, Trigger, Rate, Latency target).

---

### Task 2.2: Physical Hardware Selection and Edge Compute Allocation
* **Objective:** Finalize embedded compute nodes, sensor specifications, and execution units (CPU/GPU) for all microservices.
* **Dependencies:**
  * SAD: Section 1.0, Section 4.4, Section 5.1–Section 5.7.
  * SRS: NFR-01, NFR-03, C-01, Tier 3 Physical Validation scope[cite: 5].
* **Deliverables:**
  * HW-01: Edge compute hardware specification sheet (SoC, CPU/GPU cores, RAM, thermal design power).
  * HW-02: Sensor and networking baseline (Camera resolution/FOV/FPS, Wi-Fi ad-hoc specifications).
  * HW-03: Workload distribution matrix mapping services (C-01 to C-14) to compute targets.

---

### Task 2.3: Software Stack, AI Model Baselines, and Middleware Architecture
* **Objective:** Select concrete vision models, multi-object tracking methods, and communication middleware.
* **Dependencies:**
  * LR: Real-time edge detection, appearance Re-ID, motion compensation, decentralized coordination.
  * SAD: Section 1.0, Section 1.6, Section 3.2.
  * SRS: NFR-04, NFR-08, NFR-12[cite: 5].
* **Deliverables:**
  * SW-01: Perception stack specification (Edge-optimized YOLO with TensorRT acceleration).
  * SW-02: Visual tracker and Re-ID configuration (BoT-SORT with GMC and lightweight embeddings).
  * SW-03: Middleware architecture (ROS 2 for intra-UGV topics, MQTT for inter-UGV metadata).
  * SW-04: Technical trade-off matrix justifying framework and model selections.

---

### Task 2.4: Concrete Technical Interface and Payload Schema Definition
* **Objective:** Produce code-level schemas and validation rules for all internal and external communication interfaces.
* **Dependencies:**
  * SAD: Section 2.6, Section 3.3, Section 3.4.1, Section 3.7.
  * SRS: FR-27–FR-30, NFR-10, NFR-11[cite: 5].
* **Deliverables:**
  * IF-01: ROS 2 interface definitions (.msg/.srv) for interfaces I-01 through I-16.
  * IF-02: Inter-UGV protocol JSON schemas for all handoff messages (REQUEST, RESPONSE, COMMIT, STATUS, CONFIRM, CANCEL).
  * IF-03: Message validation specifications (Timestamping, UUID tracking, freshness checks).

---

### Task Dependency Matrix

| Task ID | Focus Area | Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- |
| Task 2.1 | Data Flow Pipelines | SAD (Sec 1, 3), SRS (FR 1-6, 14-17, 26)[cite: 5] | DF-01, DF-02, DF-03, DF-04 |
| Task 2.2 | Hardware Allocation | SAD (Sec 1, 4, 5), SRS (NFR 1, 3)[cite: 5] | HW-01, HW-02, HW-03 |
| Task 2.3 | Software & Models | LR, SAD (Sec 1, 3), SRS (NFR 4, 8, 12)[cite: 5] | SW-01, SW-02, SW-03, SW-04 |
| Task 2.4 | Interface Schemas | SAD (Sec 2, 3), SRS (FR 27-30, NFR 10, 11)[cite: 5] | IF-01, IF-02, IF-03 |
