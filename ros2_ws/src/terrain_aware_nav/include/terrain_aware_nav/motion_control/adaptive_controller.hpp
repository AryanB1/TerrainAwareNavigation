// Copyright (c) 2025 Terrain Navigation Team
// Licensed under the Apache License, Version 2.0

#ifndef TERRAIN_AWARE_NAV__ADAPTIVE_CONTROLLER_HPP_
#define TERRAIN_AWARE_NAV__ADAPTIVE_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/string.hpp"

#include "terrain_aware_nav/terrain_detection/terrain_detector.hpp"

namespace terrain_aware_nav
{

struct ControlParameters {
  double max_linear_velocity;
  double max_angular_velocity;
  double acceleration_limit;
  double deceleration_limit;
  double stability_factor;
};

class AdaptiveController
{
public:
  explicit AdaptiveController(const rclcpp::Node::SharedPtr & node);
  virtual ~AdaptiveController();

  bool initialize();
  
  geometry_msgs::msg::Twist computeVelocityCommand(
    const geometry_msgs::msg::Twist & desired_cmd,
    const TerrainType & terrain_type,
    const nav_msgs::msg::Odometry::SharedPtr & current_odom);
    
  void updateControlParameters(const TerrainType & terrain_type);

private:
  rclcpp::Node::SharedPtr node_;
  
  std::unordered_map<TerrainType, ControlParameters> control_params_map_;
  
  ControlParameters active_params_;
  
  ControlParameters default_params_;
  
  geometry_msgs::msg::Twist adaptForRoughTerrain(const geometry_msgs::msg::Twist & cmd);
  geometry_msgs::msg::Twist adaptForSteepTerrain(const geometry_msgs::msg::Twist & cmd);
  geometry_msgs::msg::Twist adaptForSlipperyTerrain(const geometry_msgs::msg::Twist & cmd);
  geometry_msgs::msg::Twist adaptForSoftTerrain(const geometry_msgs::msg::Twist & cmd);
  
  geometry_msgs::msg::Twist limitVelocity(const geometry_msgs::msg::Twist & cmd);
  
  geometry_msgs::msg::Twist last_cmd_;
};

}  // namespace terrain_aware_nav

#endif  // TERRAIN_AWARE_NAV__ADAPTIVE_CONTROLLER_HPP_