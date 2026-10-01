# 🚀 EdgeSync: Proactive TTO-Driven Decentralized Target Handoff for Dual JetAuto UGVs

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_Hawksbill-22314E?logo=ros)](https://docs.ros.org/en/humble/)
[![OS](https://img.shields.io/badge/OS-Ubuntu_22.04_LTS-E95420?logo=ubuntu)](https://ubuntu.com/)
[![Compute](https://img.shields.io/badge/Compute-NVIDIA_Jetson_Orin_Nano_8GB-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)
[![Middleware](https://img.shields.io/badge/DDS-eProsima_FastDDS_2.x-107C41)](https://fast-dds.docs.eprosima.com/)
[![Inference](https://img.shields.io/badge/Inference-TensorRT_8.6-76B900?logo=nvidia)](https://developer.nvidia.com/tensorrt)

> **Graduation Project** | **Baseline:** FROZEN & CERTIFIED (v1.0)  
> **Target Platform:** Dual Hiwonder JetAuto / JetAuto Pro UGVs  
> **Core Focus:** Real-time multi-robot target tracking, proactive occlusion prediction, and distributed consensus handoff.

---

## 📌 Project Overview

**EdgeSync** is an advanced decentralized multi-robot coordination framework designed for continuous, uninterrupted tracking of dynamic targets across cooperative Unmanned Ground Vehicles (UGVs). Leveraging edge-computed **Time-To-Occlusion (TTO)** estimation and visual Re-Identification (Re-ID), EdgeSync proactively predicts line-of-sight loss and executes an atomic target handoff to a peer robot before occlusion occurs—without relying on centralized servers or streaming raw video feeds over the network.

### 5 Non-Negotiable System Invariants
1. **Zero Dual-Ownership Invariant:** The state `COMMITTED` maintains Robot A as the sole authoritative tracker; target ownership transfers strictly upon atomic exchange of `HANDOFF_CONFIRM_ACK` with monotonic epoch increment $e \to e+1$.
2. **Deterministic Latency Budget:** Nominal execution budget $\Delta t_{exec} \le 866\,\text{ms}$, bounded by an empirical safety cutoff $\Delta t_{budget}^{max} = 1230\,\text{ms}$ to prevent target drop during critical occlusion horizons.
3. **Anti-Deadlock Liveness Invariant:** Continuous 10 Hz `ACQ_STATUS` heartbeats backed by a 300 ms watchdog timer to preemptively abort failed handoff attempts (`ERR_CANDIDATE_DEAD`) and revert immediately to solo recovery mode.
4. **Spatial Reference Consistency:** Target coordinates, velocities, and covariances are anchored in a common metric `map` frame via `tf2_ros`, with ego-motion uncertainty strictly propagated into measurement covariance matrices ($R_{total}$).
5. **Edge Resource & Resident Inference Invariant:** Deep feature extractors (512-dimensional OSNet) are pre-warmed and resident in unified memory, guaranteeing near-zero cold-start latency when handoff windows trigger.

---

## 🏗️ Repository Architecture

```
EdgeSync/
├── EdgeSync_Implementation_Plan_v1.0.md   # Master implementation execution plan (v1.0)
├── README.md                              # Main project documentation & entry point
├── Hardware Description/                  # Robot specifications and hardware manuals
│   ├── JetAuto User Manual.pdf            # Hiwonder JetAuto comprehensive manual
│   └── jetauto_technical_specifications.md# Orin Nano, STM32F407, sensor bus details
├── Market Resaerch/                       # Competitive analysis and market research
│   └── Complete Market Research.docx      # Market evaluation document
├── Software Digrams/                      # Full architectural diagrams & assets
│   ├── 1_Component_Diagram.html           # Interactive component breakdown
│   ├── 2_StateMachine_Diagram.html        # Handoff state machine lifecycle
│   ├── 3_Sequence_Diagram_Success.html    # Nominal handoff sequence
│   ├── 4_Sequence_Diagram_Failure.html    # Abort & rollback recovery sequence
│   ├── 5_Activity_Diagram.html            # End-to-end operational flow
│   ├── 6_Deployment_Diagram.html          # Embedded compute & network mapping
│   ├── 7_ROS2_Node_Graph.html             # Inter-node topics & service topology
│   ├── 8_Class_Diagram_Data_Model.html    # Data models & OOP abstractions
│   ├── 9_Timing_Diagram.html              # Bounded latency execution budget
│   ├── capture_diagrams.js                # Puppeteer script for diagram rendering
│   └── Digrams Photos/                    # High-resolution rendered PNG diagrams
│       ├── 1_Component_Diagram.png
│       ├── 2_StateMachine_Diagram.png
│       ├── 3_Sequence_Diagram_Success.png
│       ├── 4_Sequence_Diagram_Failure.png
│       ├── 5_Activity_Diagram.png
│       ├── 6_Deployment_Diagram.png
│       ├── 7_ROS2_Node_Graph.png
│       ├── 8_Class_Diagram_Data_Model.png
│       └── 9_Timing_Diagram.png
├── System Design Phase/                   # Formal engineering and academic documents
│   ├── 1_Literature_Review_Final.html     # Literature review & prior work
│   ├── 2_SRS_v1.3.html                    # Software Requirements Specification (v1.3)
│   ├── 3_SAD_v2.3.html                    # Software Architecture Document (v2.3)
│   ├── 4_Math_Protocol_ROS2_v2.3.html     # Mathematical formulations & protocol spec
│   └── 5_Experimental_Design_v1.1.html    # Evaluation & validation protocols
└── Implementation Phase/                  # ROS 2 Python implementation packages (rclpy)
    └── .gitkeep
```

---

## 📐 System Architecture & Diagrams

The following system models represent the certified architectural baseline:

| Diagram | Focus | Preview |
|:---|:---|:---:|
| **Component Diagram** | Structural breakdown of Edge AI, Tracking, and DDS subsystems | [Open Diagram](Software%20Digrams/Digrams%20Photos/1_Component_Diagram.png) |
| **State Machine** | Atomic lifecycle transitions (`MONITORING` $\to$ `COMMITTED` $\to$ `HANDOFF`) | [Open Diagram](Software%20Digrams/Digrams%20Photos/2_StateMachine_Diagram.png) |
| **Nominal Sequence** | Successful proactive target acquisition protocol | [Open Diagram](Software%20Digrams/Digrams%20Photos/3_Sequence_Diagram_Success.png) |
| **Failure Sequence** | Watchdog timeout handling, abort sequences, and rollback logic | [Open Diagram](Software%20Digrams/Digrams%20Photos/4_Sequence_Diagram_Failure.png) |
| **Deployment Model** | Distributed deployment across dual Orin Nano nodes and STM32 MCUs | [Open Diagram](Software%20Digrams/Digrams%20Photos/6_Deployment_Diagram.png) |
| **ROS 2 Node Graph** | Complete intra-robot and inter-robot communication graph | [Open Diagram](Software%20Digrams/Digrams%20Photos/7_ROS2_Node_Graph.png) |
| **Timing Budget** | Deterministic breakdown of execution latency ($\le 866\,\text{ms}$) | [Open Diagram](Software%20Digrams/Digrams%20Photos/9_Timing_Diagram.png) |

---

## ⚙️ Hardware & Software Baseline

### Hardware Specifications
* **Base Platform:** 2x Hiwonder JetAuto / JetAuto Pro (Mecanum omnidirectional chassis)
* **Primary Edge Compute:** NVIDIA Jetson Orin Nano (8 GB Unified LPDDR5, 1024-core Ampere GPU)
* **Microcontroller:** STM32F407VET6 (168 MHz ARM Cortex-M4, FreeRTOS for low-level motor PID and IMU)
* **Perception Sensors:**
  * **RGB-D Camera:** Astra Pro Plus (Monocular structured light, metric depth deprojection)
  * **LiDAR:** 2D RPLiDAR (SLAM, obstacle avoidance, local costmaps)
* **Inter-Robot Network:** Ad-Hoc 5 GHz Wi-Fi mesh / peer-to-peer DDS domain

### Software Environment
* **Operating System:** Ubuntu 22.04 LTS (Jammy Jellyfish)
* **Middleware:** ROS 2 Humble Hawksbill (strictly `rclpy` Python API for all logic)
* **Message Protocol:** FastDDS with custom `edgesync_msgs` IDL contracts
* **Neural Inference:** TensorRT 8.6 / ONNX Runtime with PyCUDA (INT8 / FP16 accelerated)
* **Vision Models:** Lightweight Real-Time Object Detector + ByteTrack MOT + OSNet Re-ID (512-dim embedding)

---

## 📄 License & Attribution

Developed by **Mustafa Mohamed** ([@Mustafa700aa](https://github.com/Mustafa700aa)) as part of the Robotics Graduation Project.  
All architectural designs, mathematical models, and implementation specifications are documented under standard academic research and open engineering guidelines.
