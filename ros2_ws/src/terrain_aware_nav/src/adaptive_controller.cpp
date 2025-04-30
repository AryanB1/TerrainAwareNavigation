#include "terrain_aware_nav/motion_control/adaptive_controller.hpp"

namespace terrain_aware_nav
{

AdaptiveController::AdaptiveController(const rclcpp::Node::SharedPtr & node)
: node_(node)
{
  // Initialize default parameters
  default_params_ = {
    .max_linear_velocity = 0.5,    // m/s
    .max_angular_velocity = 0.5,   // rad/s
    .acceleration_limit = 0.2,     // m/s²
    .deceleration_limit = 0.3,     // m/s²
    .stability_factor = 1.0        // unitless, multiplier for all params
  };
  
  // Set active params to default
  active_params_ = default_params_;
}

AdaptiveController::~AdaptiveController()
{
}

bool AdaptiveController::initialize()
{
  // Declare parameters for each terrain type
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
  
  // Get parameter values for default control params
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
  
  // Set up param map for different terrain types
  control_params_map_[TerrainType::FLAT] = default_params_;
  
  // Rough terrain params
  control_params_map_[TerrainType::ROUGH] = {
    .max_linear_velocity = node_->get_parameter("control.rough.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.rough.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.rough.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.rough.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.rough.stability_factor").as_double()
  };
  
  // Steep terrain params
  control_params_map_[TerrainType::STEEP] = {
    .max_linear_velocity = node_->get_parameter("control.steep.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.steep.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.steep.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.steep.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.steep.stability_factor").as_double()
  };
  
  // Slippery terrain params
  control_params_map_[TerrainType::SLIPPERY] = {
    .max_linear_velocity = node_->get_parameter("control.slippery.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.slippery.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.slippery.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.slippery.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.slippery.stability_factor").as_double()
  };
  
  // Soft terrain params
  control_params_map_[TerrainType::SOFT] = {
    .max_linear_velocity = node_->get_parameter("control.soft.max_linear_velocity").as_double(),
    .max_angular_velocity = node_->get_parameter("control.soft.max_angular_velocity").as_double(),
    .acceleration_limit = node_->get_parameter("control.soft.acceleration_limit").as_double(),
    .deceleration_limit = node_->get_parameter("control.soft.deceleration_limit").as_double(),
    .stability_factor = node_->get_parameter("control.soft.stability_factor").as_double()
  };
  
  // Set unknown and obstacle to default to conservative parameters
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
  
  // Initialize last command
  last_cmd_ = geometry_msgs::msg::Twist();
  
  return true;
}

void AdaptiveController::updateControlParameters(const TerrainType & terrain_type)
{
  // Check if we have parameters for this terrain type
  if (control_params_map_.find(terrain_type) != control_params_map_.end()) {
    active_params_ = control_params_map_[terrain_type];
    
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Switching to %s terrain control params (stability: %.2f, max_vel: %.2f)",
      TerrainDetector::terrainTypeToString(terrain_type).c_str(),
      active_params_.stability_factor,
      active_params_.max_linear_velocity);
  } else {
    // Fallback to default parameters
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
  // Update control parameters based on terrain
  updateControlParameters(terrain_type);
  
  // Start with the desired command
  geometry_msgs::msg::Twist cmd = desired_cmd;
  
  // Apply terrain-specific adaptations
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
      // Slow down significantly
      cmd.linear.x *= 0.3;
      cmd.angular.z *= 0.7;
      break;
    case TerrainType::FLAT:
    case TerrainType::UNKNOWN:
    default:
      // Use default parameters, no special adaptation
      break;
  }
  
  // Apply acceleration limits
  // Calculate time since last command (assuming ~10Hz control rate if no odom timestamp)
  double dt = 0.1;  // Default 10Hz control rate
  
  if (current_odom) {
    static rclcpp::Time last_time = rclcpp::Time(current_odom->header.stamp);
    rclcpp::Time current_time = rclcpp::Time(current_odom->header.stamp);
    dt = (current_time - last_time).seconds();
    if (dt <= 0.0 || dt > 1.0) {  // Sanity check
      dt = 0.1;
    }
    last_time = current_time;
  }
  
  // Apply acceleration limits
  double linear_accel = (cmd.linear.x - last_cmd_.linear.x) / dt;
  double angular_accel = (cmd.angular.z - last_cmd_.angular.z) / dt;
  
  // Check if acceleration exceeds limits
  double accel_limit = (cmd.linear.x >= last_cmd_.linear.x) ? 
                      active_params_.acceleration_limit : 
                      active_params_.deceleration_limit;
  
  if (std::abs(linear_accel) > accel_limit) {
    // Scale back to respect acceleration limit
    cmd.linear.x = last_cmd_.linear.x + std::copysign(accel_limit * dt, linear_accel);
  }
  
  // Use same logic for angular acceleration
  if (std::abs(angular_accel) > accel_limit) {
    cmd.angular.z = last_cmd_.angular.z + std::copysign(accel_limit * dt, angular_accel);
  }
  
  // Apply velocity limits
  cmd = limitVelocity(cmd);
  
  // Update last command
  last_cmd_ = cmd;
  
  return cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForRoughTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  // On rough terrain:
  // - Reduce speed to maintain stability
  // - Slightly reduce turning rate
  adapted_cmd.linear.x *= active_params_.stability_factor;
  adapted_cmd.angular.z *= (active_params_.stability_factor * 0.9);
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForSteepTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  // On steep terrain:
  // - Significantly reduce speed
  // - Maintain careful turning
  adapted_cmd.linear.x *= (active_params_.stability_factor * 0.8);
  
  // If going uphill, reduce speed further
  if (adapted_cmd.linear.x > 0) {
    adapted_cmd.linear.x *= 0.7;
  }
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForSlipperyTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  // On slippery terrain:
  // - Significantly reduce speed
  // - Reduce turning rate dramatically to prevent skidding
  adapted_cmd.linear.x *= (active_params_.stability_factor * 0.7);
  adapted_cmd.angular.z *= (active_params_.stability_factor * 0.5);
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::adaptForSoftTerrain(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist adapted_cmd = cmd;
  
  // On soft terrain (like sand, mud, tall grass):
  // - Moderate speed reduction
  // - Slightly reduce turning
  // - Consider increasing power (not modeled here)
  adapted_cmd.linear.x *= (active_params_.stability_factor * 0.8);
  adapted_cmd.angular.z *= (active_params_.stability_factor * 0.9);
  
  return adapted_cmd;
}

geometry_msgs::msg::Twist AdaptiveController::limitVelocity(
  const geometry_msgs::msg::Twist & cmd)
{
  geometry_msgs::msg::Twist limited_cmd = cmd;
  
  // Apply velocity limits
  limited_cmd.linear.x = std::clamp(
    limited_cmd.linear.x,
    -active_params_.max_linear_velocity,
    active_params_.max_linear_velocity);
  
  limited_cmd.angular.z = std::clamp(
    limited_cmd.angular.z,
    -active_params_.max_angular_velocity,
    active_params_.max_angular_velocity);
  
  // Limit other axes to zero for differential drive robots
  limited_cmd.linear.y = 0.0;
  limited_cmd.linear.z = 0.0;
  limited_cmd.angular.x = 0.0;
  limited_cmd.angular.y = 0.0;
  
  return limited_cmd;
}

}  // namespace terrain_aware_nav