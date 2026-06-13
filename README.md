# VisionFusion ADAS

A monocular vision-based Advanced Driver Assistance System (ADAS) that combines real-time object detection, multi-object tracking, monocular depth estimation, and Time-To-Collision (TTC) analysis to assess collision risk in highway driving scenarios.

The system uses a single RGB camera feed and does not require LiDAR, radar, stereo cameras, or specialized hardware.

---

## Overview

Traditional object detection systems can identify vehicles in a scene but cannot determine how far they are from the camera or whether they pose an immediate collision risk.

This project extends a standard YOLO-based detection pipeline by integrating:

* YOLOv8 for vehicle detection and tracking
* MiDaS for monocular depth estimation
* Temporal tracking for motion analysis
* Time-To-Collision (TTC) estimation
* EMA-based smoothing for stable risk assessment

The result is a lightweight perception system capable of estimating relative vehicle distance, approach velocity, and collision risk using only a single camera stream.

---

## Problem Statement

A common mistake in simple collision warning systems is assuming that larger bounding boxes imply greater danger.

This assumption suffers from scale ambiguity:

* A large truck far away may occupy a large image region.
* A motorcycle much closer may occupy a smaller image region.
* Bounding box size alone does not reliably represent real-world distance.

The initial version of this project used YOLO detections and bounding-box area thresholds to estimate risk. This approach frequently produced incorrect warnings because it lacked depth perception.

VisionFusion addresses this limitation by combining object detection with monocular depth estimation.

---

## System Architecture

### 1. Object Detection & Tracking

YOLOv8 detects highway vehicles and assigns persistent track IDs across frames.

Target classes:

| Class      | COCO ID |
| ---------- | ------- |
| Car        | 2       |
| Motorcycle | 3       |
| Bus        | 5       |
| Truck      | 7       |

Tracking allows the system to maintain object identity over time and estimate relative motion.

---

### 2. Monocular Depth Estimation

MiDaS Small generates a dense depth map from each RGB frame.

Instead of predicting absolute metric distance, MiDaS estimates relative scene depth.

The depth map is transformed into a pseudo-distance representation and queried inside each detected vehicle bounding box.

For each vehicle:

* Depth pixels inside the box are extracted
* Median depth is computed
* The value is used as a relative distance estimate

---

### 3. Time-To-Collision (TTC)

For every tracked object:

Approach Velocity = Change in Distance / Change in Time

TTC = Current Distance / Approach Velocity

A lower TTC indicates a higher collision risk.

This provides a physically meaningful warning metric compared to bounding-box size alone.

---

### 4. EMA-Based Stability Layer

One challenge encountered during development was instability in warning predictions.

Sources of noise included:

* Bounding box jitter between frames
* Depth-map fluctuations
* Small timing variations producing large velocity spikes

These effects caused warning boxes to rapidly switch between SAFE, WARNING, and CRITICAL states.

To solve this, an Exponential Moving Average (EMA) filter was introduced.

#### Distance Smoothing

The measured depth is blended with historical depth estimates.

#### Velocity Smoothing

Approach velocity is smoothed independently to suppress extreme spikes.

This produces:

* Stable TTC estimates
* Reduced false alarms
* Consistent risk visualization
* Improved user experience

---

## Risk Classification

| Level    | Condition                           |
| -------- | ----------------------------------- |
| Safe     | TTC ≥ 6 s and sufficient distance   |
| Warning  | TTC < 6 s or moderate proximity     |
| Critical | TTC < 3 s or extremely close object |

Visual indicators:

* Green → Safe
* Orange → Warning
* Red → Critical

---

## Visual Output

Each processed frame contains:

* Vehicle detections
* Persistent tracking IDs
* Estimated depth values
* Time-To-Collision estimates
* Collision risk visualization
* Processing latency statistics
* Live MiDaS depth radar (picture-in-picture)

---

## Project Evolution

### Version 1: YOLO-Based Collision Warning

Features:

* Vehicle detection
* Confidence scores
* Bounding-box area heuristic
* FPS monitoring

Limitation:

Collision risk was inferred solely from box size, leading to unreliable warnings.

---

### Version 2: VisionFusion

Enhancements:

* Monocular depth estimation
* Multi-object tracking
* TTC computation
* Relative motion analysis
* Depth radar visualization

---

### Version 3: VisionFusion (Smoothed)

Additional Improvements:

* EMA depth smoothing
* EMA velocity smoothing
* Reduced detector jitter
* Stable TTC estimates
* Consistent risk classification

This version represents the final implementation contained in this repository.

---

## Technology Stack

* Python
* OpenCV
* PyTorch
* Ultralytics YOLOv8
* MiDaS
* NumPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/VisionFusion-ADAS.git
cd VisionFusion-ADAS
```

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy ultralytics
```

---

## Running the Project

```bash
python visionfusion_engine_smoothed.py
```

The required YOLOv8 and MiDaS model weights will be downloaded automatically during first execution.

---

## Limitations

This project is a research prototype and not a production-grade ADAS.

Current limitations include:

* Monocular depth estimation provides relative depth rather than calibrated metric distance.
* Performance degrades under poor lighting, rain, fog, or heavy occlusion.
* TTC assumes smooth motion between frames.
* Detection is limited to selected vehicle classes.
* Depth estimates may become noisy for unusual scenes.
* Processing speed depends heavily on available GPU resources.

---

## Future Improvements

Potential extensions include:

* Metric depth calibration
* Lane detection integration
* Sensor fusion with radar or LiDAR
* Vehicle trajectory prediction
* Driver alert generation
* Object segmentation
* Multi-camera perception
* Real-time deployment optimization

---

## Educational Value

This project demonstrates how modern perception systems can combine:

* Semantic understanding (YOLO)
* Spatial understanding (MiDaS)
* Temporal reasoning (Tracking)
* Physics-based risk estimation (TTC)

to move from simple object detection toward a practical autonomous-driving perception pipeline.

---

## Author

Vaibhav Jain
Mechanical Engineering, IIT Delhi

Interests:
Computer Vision, Autonomous Systems, Machine Learning, and AI for Real-World Applications.

