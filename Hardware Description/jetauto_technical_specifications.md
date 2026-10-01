# Comprehensive Technical Specifications: JetAuto & JetAuto Pro

JetAuto (and its advanced variant, JetAuto Pro) is an omnidirectional ROS-based AI transport robot developed by Hiwonder. It is designed for robotics education, ROS research, and edge-AI development. 

This document provides a highly detailed breakdown of the robot's specifications, with a primary focus on its hardware architecture and electronic components.

---

## 1. Core Computing & Control Hardware

The robot utilizes a dual-tier control architecture: a high-performance Single Board Computer (SBC) for AI and ROS processing, and a microcontroller for real-time kinematic and sensory tasks.

### 1.1 ROS Main Controllers (High-Level Processing)
Users can equip the robot with different mainboards depending on their computational needs. Each board is mounted with a custom expansion board.

*   **NVIDIA Jetson Nano:** 
    *   **OS:** Ubuntu 18.04
    *   **Environment:** ROS1 Melodic (Native) / ROS2 Humble (via Docker).
    *   **Features:** Ideal for basic AI inference and standard ROS SLAM.
*   **NVIDIA Jetson Orin Nano / Orin NX (8GB/16GB):**
    *   **OS:** Ubuntu 22.04
    *   **Environment:** ROS2 Humble.
    *   **Features:** Massive leap in CUDA cores and Tensor performance. Designed for heavy edge-AI workloads, complex deep learning inference, and fast 3D mapping. Needs 12V power input.
*   **Raspberry Pi 5:**
    *   **OS:** Debian 12
    *   **Environment:** ROS2 Humble (via Docker).
    *   **Features:** Cost-effective, high CPU performance, but lacks CUDA acceleration for deep learning.
*   **Mainboard Expansion Board:** Plugs into the mainboard to provide easily accessible GPIOs, I2C ports (4-Pin), programmable buttons (KEY1 for network switching), LED status indicators, and a USB HUB communication interface.

### 1.2 Lower-Level Motion Controller (Mini STM32 Board)
This custom industrial-grade four-layer PCB handles real-time motor PID control, sensor data acquisition, and upper-level communication.

*   **Main Control Chip:** STM32F407VET6 (100-pin, ARM Cortex-M4 core, 168 MHz frequency, 512K FLASH, 192K SRAM, integrated FPU and DSP instructions).
*   **Motor Driver Chip:** YX4055 (Features built-in overcurrent and short-circuit protection).
*   **Operating System:** FreeRTOS (Embedded real-time operating system for task scheduling).
*   **Hardware Ports & Interfaces:**
    *   4x DC Encoder Motor Ports.
    *   4x Serial Bus Servo Ports.
    *   2x I2C Ports (4-PIN).
    *   1x USB HOST Port (Used for wireless PS2 controller receiver).
    *   1x USB Type-C Serial Port (For flashing firmware via SWD/ISP and communicating with the ROS Main Controller via the `RRC Communication Protocol`).
*   **Dimensions:** 78 mm x 39 mm (Mounting hole spacing: 48 x 32 mm).

---

## 2. Sensors & Perception Hardware

### 2.1 Vision Systems
*   **3D Depth Camera (Astra Pro Plus):** 
    *   **Technology:** Monocular 3D Structured Light.
    *   **Components:** Infrared (IR) camera, Infrared (IR) projector, and a dedicated depth processor.
    *   **Use Cases:** Point cloud generation, 3D spatial object grasping, RGB-D mapping, KCF object tracking, and skeletal tracking.
*   **Monocular Camera:** (Optional/Alternative configuration) Standard RGB camera for basic computer vision tasks (OpenCV, color tracking) without depth perception.

### 2.2 LiDAR (Light Detection and Ranging)
The robot is compatible with multiple 2D LiDAR models to handle SLAM (Simultaneous Localization and Mapping) and autonomous navigation.
*   **Supported Models:** RPLiDAR A1, A2, G4, S2L, and LD14P.
*   **Functionality:** 360-degree laser scanning for obstacle avoidance, dynamic path planning, LiDAR guarding, and LiDAR following.

### 2.3 Inertial & Audio Sensors
*   **IMU (Inertial Measurement Unit):** MPU6050 (Integrated directly onto the STM32 board). Features a 6-axis sensor (3-axis accelerometer + 3-axis gyroscope) for odometry and posture calculation.
*   **Microphone Array Module:** (Included in Advanced/Ultimate kits).
    *   **Hardware:** R818 noise reduction board with a circular 6-microphone array.
    *   **Features:** 360° sound reception, sound source localization, echo cancellation, background noise, and reverberation suppression.

---

## 3. Actuators & Mechanical Hardware

### 3.1 Chassis & Propulsion
*   **Motors:** 4x Hall Encoder DC Gear Motors.
    *   **Specs:** 12V operating voltage, AB phase pulse output. Provides highly accurate closed-loop PID speed feedback.
*   **Wheels:** Mecanum wheels allowing for omnidirectional movement (forward, backward, turning, and lateral/strafing movement). 
*   **Chassis Compatibility:** The firmware mathematically supports switching between Mecanum, Tracked (Tank), and Ackerman (Steering) kinematics.

### 3.2 Robotic Arm & Pan-Tilt (JetAuto Pro Only)
*   **Pan-Tilt Mechanism:** Uses 1x HTS-20H smart bus servo to pitch and yaw the depth camera.
*   **6-DOF Robotic Arm:** Constructed from durable sheet metal and highly precise smart bus servos.
    *   **Body Joints:** 3x HTD-35H servos.
    *   **Wrist Joint:** 1x HX-12H bus servo.
    *   **Gripper:** 1x HTS-21H bus servo.
    *   **Orientation:** Gripper faces the front of the chassis, visually guided by the depth camera.

---

## 4. Power Management & Interfaces

### 4.1 Battery System
*   **Type:** Lithium Polymer (LiPo) Battery.
*   **Capacity:** 6000mAh.
*   **Voltage:** 11.1V nominal (12.6V fully charged).
*   **Safety Features:** Built-in hardware protection against overcharge, over-discharge, overcurrent, and short circuits.

### 4.2 Power Distribution Module
An industrial-grade four-layer board dedicated to safe power routing.
*   **Input:** 6-14V wide voltage interface from the battery.
*   **Outputs:** 
    *   2x 5V output ports (Used to power Jetson Nano / Raspberry Pi 5).
    *   2x 12V output ports (Used to power Jetson Orin Nano / Orin NX and motors).
*   **Physical Switch:** Main push-button power switch.

### 4.3 User Interfaces & Display
*   **7-Inch LCD Touch Screen:** 
    *   **Video:** Connects via HDMI (or Micro-HDMI/DP adapters depending on the board).
    *   **Touch/Power:** Connects via USB to the `CTOUCH` interface.
*   **Wireless Controller:** PS2-style Gamepad communicating via a 2.4GHz USB receiver plugged directly into the STM32 board's USB HOST port.

---

## 5. Software & Network Specifications

### 5.1 Communication Protocols
*   **DDS (Data Distribution Service):** Used by ROS2 for decentralized, peer-to-peer node communication. (Supports tuning via Fast DDS or Cyclone DDS for multi-robot handoff algorithms).
*   **RRC Protocol:** Hiwonder’s proprietary Hexadecimal data transmission protocol used between the SBC and the STM32 over UART (Baud rate: 115200). 
    *   *Frame Structure:* Frame Header (0xAA 0x55) -> Function Code -> Data Length -> Parameters -> Checksum (CRC).

### 5.2 Network Modes
*   **AP Direct Connection Mode (Default):** The robot acts as a Wi-Fi hotspot (SSID starting with "HW"). Used for direct local control via the WonderAi App or PC without internet.
*   **STA LAN Mode:** The robot connects to a local 5G/2.4G router. Required for multi-robot collaboration, internet access, and low-latency SSH control.
*   **Fixed IP (USB):** Jetson boards support NDIS. Connecting via USB-C establishes a direct fixed IP (192.168.55.1) for highly stable headless programming.

### 5.3 Remote Development Tools
*   **NoMachine / VNC:** For full graphical desktop remote access.
*   **MobaXterm:** For lightweight SSH terminal access, executing bash/ROS commands, and X11 server graphical forwarding.

---

## 6. Highlighted AI & Functional Capabilities

*   **Autonomous Navigation & SLAM:** 2D mapping (Gmapping, Cartographer), multi-point navigation (AMCL, DWA algorithms), and dynamic obstacle avoidance.
*   **Multi-Robot Collaboration:** Features `jetauto_multi` packages for synchronized swarm mapping and fleet navigation.
*   **Computer Vision (OpenCV / Deep Learning):** Object color tracking, trash classification, and MediaPipe-based hand-gesture recognition for motion control.
*   **Augmented Reality (AR):** AprilTag detection superimposed with 3D models (e.g., bicycles) rendered in real-time on the camera feed.
*   **Voice Interaction:** Keyword wake-up ("Hello, Hiwonder") and audio-based command execution (Advanced/Ultimate kits only).
*   **Simulations:** Complete URDF models provided for Gazebo physics simulation and MoveIt kinematic path planning for the robotic arm.