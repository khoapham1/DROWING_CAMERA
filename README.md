# UAV Drowning Detection and Rescue Support System

A real-time UAV-based drowning detection and rescue support platform that combines **DroneKit/Pixhawk flight control**, **YOLOv5 person detection**, **MediaPipe pose estimation**, and a **Flask + Leaflet ground control dashboard** for aerial monitoring and emergency localization.

## Overview
This project is designed for UAV-assisted water safety monitoring. The drone streams live video, detects people in water, analyzes distress gestures, and reports GPS coordinates of SOS events to the ground station interface.

## Key Features
- Real-time aerial video streaming from onboard camera
- Person detection using **YOLOv5 ONNX**
- Drowning / SOS state analysis using **MediaPipe Pose**
- GPS-tagged emergency reporting
- Web dashboard with live map, telemetry, mission controls, and recording tools
- Drone mission control through **DroneKit + MAVLink + Pixhawk**

## System Architecture
- **Frontend:** HTML dashboard + Leaflet map + Socket.IO
- **Backend:** Flask + Flask-SocketIO server
- **Flight Control:** DroneKit, pymavlink, Pixhawk
- **Perception:** YOLOv5 ONNX, MediaPipe Pose
- **Detection Logic:** custom drowning state machine (`ACTIVE -> FROZEN -> SOS`)

## Repository Structure
```text
DROWING_CAMERA/
├── static/                    # icons and static assets
├── templates/
│   └── index.html            # web dashboard UI
├── server.py                 # Flask + SocketIO control server
├── drone_control.py          # UAV controller, telemetry, mission logic
├── drowing_detector.py       # drowning / distress detection logic
├── person_detector.py        # person detection model wrapper
├── planner.py                # waypoint planning utilities
├── PID_controller.py         # control helper
├── file_gps_station.json     # GPS waypoint presets
├── test_camera.py            # camera test
├── test_detect_person.py     # person detection test
├── test_drowng.py            # drowning detection test
├── camera_matrix_logitech.npy
├── dist_coeff_logitech.npy
└── yolov5n_quant.onnx
```

## Workflow
1. Start camera stream and flight controller connection
2. Run person detection on incoming frames
3. Apply pose estimation for body landmarks
4. Evaluate distress gestures and state transitions
5. Push GPS-tagged SOS events to the ground dashboard
6. Visualize detections, telemetry, and mission status in real time

## Technologies
- Python
- Flask / Flask-SocketIO
- OpenCV
- DroneKit
- pymavlink
- MediaPipe
- YOLOv5 ONNX
- Leaflet.js

## Safety Note
This repository is intended for **research, simulation, and prototype development** in UAV-assisted rescue monitoring. Field deployment should include geofencing, manual override, failsafe logic, communication redundancy, and regulatory compliance.

## Future Improvements
- Multi-target tracking with persistent IDs
- Thermal camera integration for low-visibility water rescue
- Mission planner integration with autonomous search patterns
- Edge AI optimization on Raspberry Pi / Jetson
- Event logging and post-mission analysis dashboard
