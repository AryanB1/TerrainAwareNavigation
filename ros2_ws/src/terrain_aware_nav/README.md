# Terrain-Aware Navigation

A ROS 2 package that enables robots to adapt their movement behavior based on detected terrain conditions.

## Overview

This package provides a system that:
1. Detects and classifies terrain types using sensor data (LiDAR, camera, IMU)
2. Adapts the robot's motion control parameters based on the detected terrain
3. Publishes terrain information and adjusted velocity commands

## Features

- Multi-sensor terrain detection and classification
- Support for various terrain types: flat, rough, steep, slippery, soft, obstacles
- Adaptive motion control for different terrain conditions
- Configurable parameters for fine-tuning terrain detection and control

## Supported Terrain Types

- **FLAT**: Hard, even surfaces (e.g., pavement, indoor floors)
- **ROUGH**: Uneven but navigable terrain (e.g., gravel, bumpy ground)
- **STEEP**: Inclines and slopes that require reduced speed
- **SLIPPERY**: Low-friction surfaces (e.g., wet/icy surfaces)
- **SOFT**: Deformable terrain (e.g., sand, tall grass, mud)
- **OBSTACLE**: Areas to avoid or approach with extreme caution

## Prerequisites

- ROS 2 (tested on Foxy and newer)
- PCL (Point Cloud Library)
- OpenCV
- Robot with LiDAR and/or camera and IMU sensors

## Installation

```bash
# Clone the repository into your ROS 2 workspace source directory
git clone https://github.com/yourusername/terrain_aware_nav.git ~/your_workspace/src/

# Build the package
cd ~/your_workspace
colcon build --packages-select terrain_aware_nav

# Source the workspace
source install/setup.bash
```

## Usage

1. Configure the parameters in `config/terrain_params.yaml` for your specific robot and sensors.

2. Launch the terrain navigation node:
```bash
ros2 launch terrain_aware_nav terrain_navigation.launch.py
```

3. The node will subscribe to sensor topics and listen to raw velocity commands from `/cmd_vel_raw`. It will then publish adapted velocity commands to `/cmd_vel` and terrain information to `/terrain_type`.

## Configuration

The system behavior can be customized via the parameters in `config/terrain_params.yaml`:

- Adjust terrain detection thresholds
- Configure motion control parameters for each terrain type
- Set topic names for sensors and control
- Enable/disable debug mode

## Topics

### Subscriptions
- `/camera/image_raw` - Camera images for visual terrain classification
- `/lidar/points` - Point cloud data for terrain geometry analysis
- `/imu/data` - IMU readings for slope and vibration analysis
- `/odom` - Odometry for velocity and position tracking
- `/cmd_vel_raw` - Raw velocity commands to be adapted

### Publications
- `/cmd_vel` - Adapted velocity commands based on terrain
- `/terrain_type` - Current detected terrain type as a string

## License

Apache License 2.0