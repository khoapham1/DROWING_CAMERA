# UAV Drowning Detection and Rescue Support System

A real-time UAV-based drowning detection and rescue support platform that combines **DroneKit/Pixhawk flight control**, **YOLOv5 person detection**, **MediaPipe pose estimation**, and a **Flask + Leaflet ground control dashboard** for aerial monitoring and emergency localization.

## Overview
This project is designed for UAV-assisted water safety monitoring. The drone streams live video, detects people in water, analyzes distress gestures, and reports GPS coordinates of SOS events to the ground station interface.
<img width="736" height="521" alt="image" src="https://github.com/user-attachments/assets/310a23b4-d553-43c4-b5ff-70f7f593dcbf" />



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
<img width="962" height="579" alt="image" src="https://github.com/user-attachments/assets/62b3a2c8-fad1-4866-b4fa-97e250e7582c" />

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
# GitHub Makeover Notes for DROWING_CAMERA

## 1. Rename for professionalism
Current name: `DROWING_CAMERA`
Recommended:
- `UAV-Drowning-Detection-System`
- `Drone-Drowning-Rescue-Support`
- `Vision-Based-UAV-Water-Rescue`

## 2. Fix naming consistency
- `drowing_detector.py` -> `drowning_detector.py`
- `test_drowng.py` -> `test_drowning.py`
- `get_lastest_frame()` -> `get_latest_frame()`

## 3. Improve About section
Description:
`Real-time UAV drowning detection and rescue support system using YOLOv5, MediaPipe Pose, DroneKit, MAVLink, and a web-based ground control dashboard.`

Topics:
`uav, drone, dronekit, mavlink, pixhawk, yolov5, mediapipe, computer-vision, rescue-drone, drowning-detection, flask, telemetry, raspberry-pi`

## 4. Better root structure
```text
project/
├── app/
│   ├── server.py
│   ├── drone_control.py
│   ├── drowning_detector.py
│   ├── person_detector.py
│   └── planner.py
├── templates/
├── static/
├── models/
│   └── yolov5n_quant.onnx
├── calibration/
│   ├── camera_matrix_logitech.npy
│   └── dist_coeff_logitech.npy
├── config/
│   └── file_gps_station.json
├── tests/
│   ├── test_camera.py
│   ├── test_detect_person.py
│   └── test_drowning.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 5. What makes it look like a UAV engineer repo
- Show system architecture block diagram
- Show mission/control/perception pipeline
- Add hardware stack: Pixhawk, Raspberry Pi, USB camera, telemetry link
- Add safety/failsafe note
- Add telemetry fields and map interface screenshots
- Add real deployment limitations and future roadmap
- <img width="392" height="462" alt="image" src="https://github.com/user-attachments/assets/d3fa645b-9cd6-44ee-8927-d1f093fefb30" />


## 6. Must-have README sections
- Project overview
- Features
- Hardware and software stack
- System architecture
- Setup and run
- Web dashboard preview
- Detection logic
- Mission workflow
- Limitations
- Future work
- <img width="800" height="399" alt="image" src="https://github.com/user-attachments/assets/b2399f77-a7a6-4eb7-8bbd-4c90f05560fb" />
<img width="793" height="234" alt="image" src="https://github.com/user-attachments/assets/d3bca72e-20b3-4086-8199-a83057d57239" />



## 7. Nice finishing touches
- Add badges (Python, Flask, OpenCV, DroneKit)
- Add demo GIF or screenshots
- Add `requirements.txt`
- Add `LICENSE`
- Add `.gitignore`
- Create one release tag like `v1.0.0-prototype`
