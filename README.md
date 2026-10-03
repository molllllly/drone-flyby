# drone-flyby
# Nordic AI Cup 2026 — Drone Flyby Tracking

This repository contains my computer vision solution for the **Drone Flyby** task from the Nordic AI Cup 2026.

The challenge involved detecting small objects in aerial video under limited resolution and changing viewpoints.

## Approach

Instead of treating every frame independently, I explored whether temporal information could make weak detections more stable.

The pipeline combines:

* **YOLO** for low-confidence object proposals
* **Lucas–Kanade optical flow** for estimating frame-to-frame motion
* **RANSAC affine transformation** for robust global motion estimation
* **Bounding-box propagation** across frames
* **Temporal confirmation** to filter unstable detections
* **Class voting** to make object classification more consistent over time

The system stays at the full Level-0 view and uses motion information from consecutive frames to maintain object tracks.

## Motivation

Small objects were often detected with low confidence. Rather than immediately discarding these detections, I tested whether repeated observations and geometric consistency across frames could provide stronger evidence.

The basic idea was:

```text
YOLO proposals
      ↓
Optical flow
      ↓
Affine motion estimation
      ↓
Track propagation
      ↓
Temporal confirmation
      ↓
Class voting
```

## Result

The tracking pipeline improved temporal consistency and provided a useful experiment in combining object detection with classical computer vision.

However, the hidden evaluation showed that tracking alone could not fully compensate for the domain gap between the training data and the hidden test environment.

The main lesson from the project was that **robust detector generalization was a larger bottleneck than temporal localization**.

## Technologies

Python, PyTorch, Ultralytics YOLO, OpenCV, NumPy

## File

`example.py` contains the complete inference and tracking pipeline.
