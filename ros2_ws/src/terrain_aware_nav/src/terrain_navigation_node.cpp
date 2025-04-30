#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/string.hpp"

#include "terrain_aware_nav/terrain_detection/terrain_detector.hpp"
#include "terrain_aware_nav/terrain_classification/terrain_classifier.hpp"
#include "terrain_aware_nav/motion_control/adaptive_controller.hpp"

using namespace std::chrono_literals;

namespace terrain_aware_nav
{

class TerrainNavigationNode : public rclcpp::Node
{
public:
  TerrainNavigationNode()
  : Node("terrain_navigation_node")
  {
    // Declare parameters
    declare_parameter("camera_topic", "/camera/image_raw");
    declare_parameter("lidar_topic", "/lidar/points");
    declare_parameter("imu_topic", "/imu/data");
    declare_parameter("odom_topic", "/odom");
    declare_parameter("cmd_vel_topic", "/cmd_vel");
    declare_parameter("terrain_topic", "/terrain_type");
    declare_parameter("control_rate", 10.0);  // Hz
    declare_parameter("debug_mode", false);
    
    // Get parameters
    camera_topic_ = get_parameter("camera_topic").as_string();
    lidar_topic_ = get_parameter("lidar_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    terrain_topic_ = get_parameter("terrain_topic").as_string();
    double control_rate = get_parameter("control_rate").as_double();
    debug_mode_ = get_parameter("debug_mode").as_bool();
    
    // Create components
    terrain_detector_ = std::make_shared<TerrainDetector>(shared_from_this());
    terrain_classifier_ = std::make_shared<TerrainClassifier>(shared_from_this());
    adaptive_controller_ = std::make_shared<AdaptiveController>(shared_from_this());
    
    // Initialize components
    terrain_detector_->initialize();
    terrain_classifier_->initialize();
    adaptive_controller_->initialize();
    
    // Create publishers
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, 10);
    terrain_pub_ = create_publisher<std_msgs::msg::String>(
      terrain_topic_, 10);
    
    // Create subscribers
    camera_sub_ = create_subscription<sensor_msgs::msg::Image>(
      camera_topic_, 10,
      std::bind(&TerrainNavigationNode::cameraCallback, this, std::placeholders::_1));
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, 10,
      std::bind(&TerrainNavigationNode::lidarCallback, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, 10,
      std::bind(&TerrainNavigationNode::imuCallback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&TerrainNavigationNode::odomCallback, this, std::placeholders::_1));
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel_raw", 10,
      std::bind(&TerrainNavigationNode::cmdVelCallback, this, std::placeholders::_1));
    
    // Create control timer
    control_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / control_rate),
      std::bind(&TerrainNavigationNode::controlLoop, this));
    
    RCLCPP_INFO(
      get_logger(),
      "Terrain Navigation Node initialized with control rate: %.1f Hz", control_rate);
  }
  
private:
  // Component instances
  std::shared_ptr<TerrainDetector> terrain_detector_;
  std::shared_ptr<TerrainClassifier> terrain_classifier_;
  std::shared_ptr<AdaptiveController> adaptive_controller_;

  // ROS Parameters
  std::string camera_topic_;
  std::string lidar_topic_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string cmd_vel_topic_;
  std::string terrain_topic_;
  bool debug_mode_;
  
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr terrain_pub_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

  // Timer
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Sensor data cache
  sensor_msgs::msg::Image::SharedPtr latest_camera_data_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_lidar_data_;
  sensor_msgs::msg::Imu::SharedPtr latest_imu_data_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_data_;
  geometry_msgs::msg::Twist latest_cmd_vel_;
  
  // Current terrain state
  TerrainType current_terrain_type_{TerrainType::UNKNOWN};
  
  // Callback functions
  void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    latest_camera_data_ = msg;
  }
  
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    latest_lidar_data_ = msg;
  }
  
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    latest_imu_data_ = msg;
  }
  
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_data_ = msg;
  }
  
  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    latest_cmd_vel_ = *msg;
  }
  
  // Main control loop
  void controlLoop()
  {
    // Skip if we don't have the necessary data yet
    if (!latest_lidar_data_ || !latest_camera_data_ || !latest_imu_data_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,  // Throttle to log every 5 seconds
        "Waiting for sensor data...");
      return;
    }
    
    // 1. Detect terrain
    TerrainType detected_terrain = terrain_detector_->detectTerrain(
      latest_lidar_data_, latest_camera_data_, latest_imu_data_);
    
    // 2. Extract terrain features for more detailed classification
    TerrainFeatures terrain_features = terrain_classifier_->extractFeatures(
      latest_lidar_data_, latest_camera_data_);
    
    // 3. Classify terrain with more detail
    TerrainType classified_terrain = terrain_classifier_->classifyTerrain(terrain_features);
    
    // If detection and classification agree, use that classification
    // Otherwise use the one with higher confidence
    double detection_confidence = terrain_detector_->getTerrainConfidence();
    double classification_confidence = terrain_classifier_->getClassificationConfidence();
    
    if (detected_terrain == classified_terrain) {
      current_terrain_type_ = detected_terrain;
    } else if (classification_confidence > detection_confidence) {
      current_terrain_type_ = classified_terrain;
    } else {
      current_terrain_type_ = detected_terrain;
    }
    
    // Publish current terrain type
    auto terrain_msg = std::make_unique<std_msgs::msg::String>();
    terrain_msg->data = TerrainDetector::terrainTypeToString(current_terrain_type_);
    terrain_pub_->publish(std::move(terrain_msg));
    
    // Log terrain info if in debug mode
    if (debug_mode_) {
      RCLCPP_INFO(
        get_logger(),
        "Terrain: %s (confidence: %.2f)",
        TerrainDetector::terrainTypeToString(current_terrain_type_).c_str(),
        std::max(detection_confidence, classification_confidence));
    }
    
    // 4. Apply adaptive control based on terrain
    if (latest_cmd_vel_.linear.x != 0.0 || latest_cmd_vel_.angular.z != 0.0) {
      geometry_msgs::msg::Twist adapted_cmd = adaptive_controller_->computeVelocityCommand(
        latest_cmd_vel_, current_terrain_type_, latest_odom_data_);
      
      // Publish adapted velocity command
      cmd_vel_pub_->publish(adapted_cmd);
      
      if (debug_mode_) {
        RCLCPP_INFO(
          get_logger(),
          "Original cmd: [%.2f, %.2f] -> Adapted: [%.2f, %.2f] for %s terrain",
          latest_cmd_vel_.linear.x, latest_cmd_vel_.angular.z,
          adapted_cmd.linear.x, adapted_cmd.angular.z,
          TerrainDetector::terrainTypeToString(current_terrain_type_).c_str());
      }
    }
  }
};

}  // namespace terrain_aware_nav

// Main function
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<terrain_aware_nav::TerrainNavigationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}