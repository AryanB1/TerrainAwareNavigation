# Terrain-Aware Navigation

ROS 2 package that adapts robot movement based on terrain conditions.

## What it does

- Detects terrain types using sensor fusion (LiDAR, camera, IMU)
- Adjusts motion control parameters in real-time
- Handles varied terrain: flat, rough, steep, slippery, soft surfaces and obstacles

## Requirements

- ROS 2 (Foxy+)
- PCL and OpenCV
- Robot with LiDAR/camera and IMU
