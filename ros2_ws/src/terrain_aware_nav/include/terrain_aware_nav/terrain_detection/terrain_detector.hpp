// Copyright (c) 2025 Terrain Navigation Team
// Licensed under the Apache License, Version 2.0

#ifndef TERRAIN_AWARE_NAV__TERRAIN_DETECTOR_HPP_
#define TERRAIN_AWARE_NAV__TERRAIN_DETECTOR_HPP_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/string.hpp"
#include "cv_bridge/cv_bridge.h"

namespace terrain_aware_nav
{

enum class TerrainType {
  UNKNOWN,
  FLAT,
  ROUGH,
  STEEP,
  SLIPPERY,
  SOFT,
  OBSTACLE
};

class TerrainDetector
{
public:
  explicit TerrainDetector(const rclcpp::Node::SharedPtr & node);
  virtual ~TerrainDetector();

  // Initialize the detector with parameters
  bool initialize();
  
  // Process sensor data to detect terrain
  TerrainType detectTerrain(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_data,
    const sensor_msgs::msg::Image::ConstSharedPtr & camera_data,
    const sensor_msgs::msg::Imu::ConstSharedPtr & imu_data);
  
  // Get the terrain confidence level
  double getTerrainConfidence() const;

  // Get string representation of terrain type
  static std::string terrainTypeToString(TerrainType type);

private:
  // Node handle
  rclcpp::Node::SharedPtr node_;
  
  // Parameters
  double slope_threshold_;
  double roughness_threshold_;
  
  // Detection methods
  TerrainType detectFromPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  TerrainType detectFromImage(const sensor_msgs::msg::Image::ConstSharedPtr & image);
  TerrainType detectFromIMU(const sensor_msgs::msg::Imu::ConstSharedPtr & imu_data);
  
  // Terrain detection confidence (0.0 to 1.0)
  double terrain_confidence_;
  
  // Last detected terrain type
  TerrainType current_terrain_type_;
};

}  // namespace terrain_aware_nav

#endif  // TERRAIN_AWARE_NAV__TERRAIN_DETECTOR_HPP_