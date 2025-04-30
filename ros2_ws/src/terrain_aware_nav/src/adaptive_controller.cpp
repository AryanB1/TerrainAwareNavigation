#include "terrain_aware_nav/motion_control/adaptive_controller.hpp"

namespace terrain_aware_nav
{

AdaptiveController::AdaptiveController(const rclcpp::Node::SharedPtr & node)
: node_(node)
{
  default_params_ = {
    .max_linear_velocity = 0.5,
    .max_angular_velocity = 0.5,
    .acceleration_limit = 0.2,
    .deceleration_limit = 0.3,
    .stability_factor = 1.0
  };
  
  active_params_ = default_params_;
}

AdaptiveController::~AdaptiveController()
{
}

bool AdaptiveController::initialize()
{
  node_->declare_parameter("control.flat.max_linear_velocity", 0.5);
  node_->declare_parameter("control.flat.max_angular_velocity", 0.5);
  node_->declare_parameter("control.flat.acceleration_limit", 0.2);
  node_->declare_parameter("control.flat.deceleration_limit", 0.3);
  node_->declare_parameter("control.flat.stability_factor", 1.0);
  
  node_->declare_parameter("control.rough.max_linear_velocity", 0.3);
  node_->declare_parameter("control.rough.max_angular_velocity", 0.3);
  node_->declare_parameter("control.rough.acceleration_limit", 0.15);
  node_->declare_parameter("control.rough.deceleration_limit", 0.25);
  node_->declare_parameter("control.rough.stability_factor", 0.7);
  
  node_->declare_parameter("control.steep.max_linear_velocity", 0.2);
  node_->declare_parameter("control.steep.max_angular_velocity", 0.2);
  node_->declare_parameter("control.steep.acceleration_limit", 0.1);
  node_->declare_parameter("control.steep.deceleration_limit", 0.3);
  node_->declare_parameter("control.steep.stability_factor", 0.5);
  
  node_->declare_parameter("control.slippery.max_linear_velocity", 0.2);
  node_->declare_parameter("control.slippery.max_angular_velocity", 0.15);
  node_->declare_parameter("control.slippery.acceleration_limit", 0.05);
  node_->declare_parameter("control.slippery.deceleration_limit", 0.1);
  node_->declare_parameter("control.slippery.stability_factor", 0.5);
  
  node_->declare_parameter("control.soft.max_linear_velocity", 0.3);
  node_->declare_parameter("control.soft.max_angular_velocity", 0.3);
  node_->declare_parameter("control.soft.acceleration_limit", 0.15);
  node_->declare_parameter("control.soft.deceleration_limit", 0.2);
  node_->declare_parameter("control.soft.stability_factor", 0.8);
  
  default_params_.max_linear_velocity = node_->get_parameter(
    "control.flat.max_linear_velocity").as_double();
  default_params_.max_angular_velocity = node_->get_parameter(
    "control.flat.max_angular_velocity").as_double();
  default_params_.acceleration_limit = node_->get_parameter(
    "control.flat.acceleration_limit").as_double();
  default_params_.deceleration_limit = node_->get_parameter(
    "control.flat.deceleration_limit").as_double();
  default_params_.stability_factor = node_->get_parameter(
    "control.flat.stability_factor").as_double();
  
  control_params_map_[TerrainType::FLAT] = default_params_;
  
  control_params_map_[TerrainType::ROUGH] = {
    .max_linear_velocity = node_->get_parameter("control.rough.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.rough.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.rough.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.rough.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.rough.stability_factor").as_double()
  };
  
  control_params_map_[TerrainType::STEEP] = {
    .max_linear_velocity = node_->get_parameter("control.steep.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.steep.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.steep.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.steep.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.steep.stability_factor").as_double()
  };
  
  control_params_map_[TerrainType::SLIPPERY] = {
    .max_linear_velocity = node_->get_parameter("control.slippery.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.slippery.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.slippery.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.slippery.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.slippery.stability_factor").as_double()
  };
  
  control_params_map_[TerrainType::SOFT] = {
    .max_linear_velocity = node_->get_parameter("control.soft.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.soft.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.soft.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.soft.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.soft.stability_factor").as_double()
  };
  
  control_params_map_[TerrainType::UNKNOWN] = control_params_map_[TerrainType::ROUGH];
  control_params_map_[TerrainType::OBSTACLE] = {
    .max_linear_velocity = 0.1,
    .max_angular_velocity = 0.2,
    .acceleration_limit = 0.05,
    .deceleration_limit = 0.4,
    .stability_factor = 0.4
  };
  
  RCLCPP_INFO(
    node_->get_logger(),
    "AdaptiveController initialized with %zu terrain parameter sets",
    control_params_map_.size());
  
  last_cmd_ = geometry_msgs::msg::Twist();
  
  return true;
}

void AdaptiveController::updateControlParameters(const TerrainType & terrain_type)
{
  if (control_params_map_.find(terrain_type) != control_params_map_.end()) {
    active_params_ = control_params_map_[terrain_type];
    
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Switching to %s terrain control params (stability: %.2f, max_vel: %.2f)",
      TerrainDetector::terrainTypeToString(terrain_type).c_str(),
      active_params_.stability_factor,
      active_params_.max_linear_velocity);
  } else {
    active_params_ = default_params_;
    
    RCLCPP_WARN(
      node_->get_logger(),
      "No control parameters for terrain type %s, using defaults",
      TerrainDetector::terrainTypeToString(terrain_type).c_str());
  }
}

geometry_msgs::msg::Twist AdaptiveController::computeVelocityCommand(
  const geometry_msgs::msg::Twist & desired_cmd,
  const TerrainType & terrain_type,
  const nav_msgs::msg::Odometry::SharedPtr & current_odom)
{
  updateControlParameters(terrain_type);
  
  geometry_msgs::msg::Twist cmd = desired_cmd;
  
  switch (terrain_type) {
    case TerrainType::ROUGH:
      cmd = adaptForRoughTerrain(cmd);
      break;
    case TerrainType::STEEP:
      cmd = adaptForSteepTerrain(cmd);
      break;
    case TerrainType::SLIPPERY:
      cmd = adaptForSlipperyTerrain(cmd);
      break;
    case TerrainType::SOFT:
      cmd = adaptForSoftTerrain(cmd);
      break;
    case TerrainType::OBSTACLE:
      cmd.linear.x *= 0.3;
      cmd.angular.z *= 0.7;
      break;
    case TerrainType::FLAT:
    case TerrainType::UNKNOWN:
    default:
      break;
  }
  
  double dt = 0.1;
  
  if (current_odom) {
    static rclcpp::Time last_time = rclcpp::Time(current_odom->header.stamp);
    rclcpp::Time current_time = rclcpp::Time(current_odom->header.stamp);
    dt = (current_time - last_time).seconds();
    if (dt <= 0.0 || dt > 1.0) {
      dt = 0.1;
    }
    last_time = current_time;
  }
  
  double linear_accel = (cmd.linear.x - last_cmd_.linear.x) / dt;
  double angular_accel = (cmd.angular.z - last_cmd_.angular.z) / dt;
  
  double accel_limit = (cmd.linear.x >= last_cmd_.linear.x) ? 
                      active_params_.acceleration_limit : 
                      active_params_.deceleration_limit;
  
  if (std::abs(linear_accel) > accel_limit) {
    cmd.linear.x = last_cmd_.linear.x + std::copysign(accel_limit * dt, linear_accel);
  }
  
  if (std::abs(angular_accel) > accel_limit) {
    cmd.angular.z = last_cmd_.angular.z + std::copysign(accel_limit * dt, angular_accel);
  }
  
  cmd = limitVelocity(cmd);
  
  last_cmd_ = cmd;
  
  return cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForRoughTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  adapted_cmd.linear.x *= active_params_.stability_factor;
  adapted_cmd.angular.z *= (active_params_.stability_factor * 0.9);
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForSteepTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  adapted_cmd.linear.x *= (active_params_.stability_factor * 0.8);
  
  if (adapted_cmd.linear.x > 0) {
    adapted_cmd.linear.x *= 0.7;
  }
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForSlipperyTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  adapted_cmd.linear.x *= (active_params_.stability_factor * 0.7);
  adapted_cmd.angular.z *= (active_params_.stability_factor * 0.5);
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForSoftTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  adapted_cmd.linear.x *= (active_params_.stability_factor * 0.8);
  adapted_cmd.angular.z *= (active_params_.stability_factor * 0.9);
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::limitVelocity(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist limited_cmd = cmd;
  
  limited_cmd.linear.x = std::clamp(
    limited_cmd.linear.x,
    -active_params_.max_linear_velocity,
    active_params_.max_linear_velocity);
  
  limited_cmd.angular.z = std::clamp(
    limited_cmd.angular.z,
    -active_params_.max_angular_velocity,
    active_params_.max_angular_velocity);
  
  limited_cmd.linear.y = 0.0;
  limited_cmd.linear.z = 0.0;
  limited_cmd.angular.x = 0.0;
  limited_cmd.angular.y = 0.0;
  
  return limited_cmd;
}

}  // namespace terrain_aware_nav