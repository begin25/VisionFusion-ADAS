# VisionFusion ADAS

### Monocular Vision-Based Collision Risk Assessment using YOLOv8, MiDaS, Tracking, and Time-To-Collision Estimation

![VisionFusion Demo](assets/visionfusion_demo.png)

**VisionFusion ADAS** is a computer vision system that transforms a standard monocular camera into a lightweight Advanced Driver Assistance System (ADAS).

The project combines real-time object detection, multi-object tracking, monocular depth estimation, and Time-To-Collision (TTC) analysis to estimate collision risk in highway driving scenarios without requiring LiDAR, radar, stereo cameras, or specialized hardware.

---

## Demo Video

https://youtu.be/VMLkfBw-CnQ

The demo demonstrates:

* Real-time vehicle detection
* Persistent object tracking
* Monocular depth estimation
* Time-To-Collision prediction
* Dynamic collision-risk assessment
* Live depth radar visualization

---

## Motivation

Traditional object detection systems can identify vehicles but cannot determine how far they are from the camera or whether they pose an immediate collision threat.

My initial implementation relied solely on YOLO detections and bounding-box size to estimate danger. This approach suffered from a fundamental limitation:

### Scale Ambiguity

A large truck far away may occupy a larger image region than a nearby motorcycle.

As a result, bounding-box size alone is not a reliable indicator of collision risk.

To address this problem, VisionFusion introduces a depth-aware perception pipeline that combines object detection, monocular depth estimation, and temporal reasoning.

---

## Key Features

### Vehicle Detection

* YOLOv8 Nano
* Real-time inference
* Vehicle-specific filtering
* Cars
* Motorcycles
* Buses
* Trucks

### Multi-Object Tracking

Persistent object IDs are maintained across frames using YOLO tracking mode.

Tracking enables:

* Motion estimation
* Relative velocity computation
* Time-To-Collision analysis

### Monocular Depth Estimation

MiDaS Small generates a dense depth map from a single RGB image.

The system extracts depth information within each detected vehicle bounding box to estimate relative distance from the ego vehicle.

### Time-To-Collision (TTC)

Collision risk is estimated using:

TTC = Distance / Relative Approach Velocity

Rather than relying on object size, the system evaluates how quickly the distance between vehicles is shrinking.

### Live Depth Radar

A picture-in-picture depth visualization displays the scene's estimated depth structure in real time.

### Collision Risk Assessment

Objects are categorized into three risk levels:

| Risk Level | Condition |
| ---------- | --------- |
| Safe       | TTC ≥ 6 s |
| Warning    | TTC < 6 s |
| Critical   | TTC < 3 s |

Visual indicators:

* Green → Safe
* Orange → Warning
* Red → Critical

---

## Engineering Challenge: The "Disco Box" Problem

After integrating YOLO and MiDaS, a new issue emerged.

Bounding boxes frequently oscillated between risk levels despite the scene appearing stable.

Root causes included:

* Bounding-box jitter
* Noisy depth estimates
* Small frame-to-frame timing variations
* Velocity spikes caused by depth fluctuations

This produced unstable warnings and poor user experience.

---

## Solution: EMA-Based Temporal Stabilization

To suppress noisy measurements, an Exponential Moving Average (EMA) filter was introduced.

### Distance Smoothing

Depth measurements are blended with historical estimates.

### Velocity Smoothing

Relative velocity estimates are independently smoothed to prevent extreme TTC spikes.

Benefits:

* Stable warning boxes
* Reduced false alarms
* Consistent TTC estimation
* Improved visual clarity
* Better temporal robustness

This stabilization layer represents the final iteration of the project.

---

## System Architecture

### Stage 1: Object Detection & Tracking

YOLOv8 identifies vehicles and assigns persistent track IDs.

### Stage 2: Depth Estimation

MiDaS generates a dense monocular depth map.

### Stage 3: Sensor Fusion

Detected vehicle regions are mapped onto the depth image.

Median depth values are extracted for each tracked object.

### Stage 4: Temporal Analysis

Historical depth measurements are used to estimate approach velocity.

### Stage 5: Time-To-Collision Estimation

TTC is computed using relative distance and closing speed.

### Stage 6: Risk Visualization

Objects are color-coded based on collision risk.

---

## Project Evolution

### Version 1 — YOLO-Based Collision Warning

Features:

* Vehicle detection
* Confidence scores
* Bounding-box area heuristic
* FPS monitoring

Limitations:

* No depth perception
* No tracking
* Frequent false warnings
* Scale ambiguity problem

---

### Version 2 — VisionFusion

Added:

* MiDaS depth estimation
* Multi-object tracking
* TTC computation
* Depth radar visualization

Result:

* Spatial awareness
* Motion understanding
* More realistic collision assessment

---

### Version 3 — VisionFusion (Final)

Added:

* EMA distance smoothing
* EMA velocity smoothing
* Stable TTC estimates
* Robust warning visualization

Result:

* Significantly improved stability
* Reduced warning oscillations
* Better user experience

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
git clone https://github.com/yourusername/VisionFusion-ADAS.git
cd VisionFusion-ADAS
```

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy ultralytics
```

---

## Run

```bash
python visionfusion_engine_smoothed.py
```

Model weights will be downloaded automatically during the first execution.

---

## Limitations

* Monocular depth is relative rather than metric.
* Performance may degrade in poor lighting conditions.
* TTC assumes smooth object motion.
* Detection is limited to selected vehicle classes.
* Real-world deployment would require calibrated sensors and extensive validation.

---

## Future Work

* Metric depth calibration
* Lane detection
* Sensor fusion with radar or LiDAR
* Vehicle trajectory prediction
* Driver alert generation
* Edge-device optimization
* Real-time deployment

---

## Results

The final system provides:

* Semantic understanding through object detection
* Spatial understanding through depth estimation
* Temporal understanding through tracking
* Physics-based collision prediction through TTC

Together, these components transform a standard monocular camera into a lightweight ADAS perception pipeline.

---

## Author

**Vaibhav Jain**

B.Tech, Mechanical Engineering
Indian Institute of Technology Delhi

Interests:

* Computer Vision
* Autonomous Systems
* Machine Learning
* Robotics
* AI for Transportation
* Robotics
