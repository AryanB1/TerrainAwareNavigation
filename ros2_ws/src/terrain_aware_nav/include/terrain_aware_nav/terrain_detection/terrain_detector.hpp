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

  bool initialize();
  
  TerrainType detectTerrain(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_data,
    const sensor_msgs::msg::Image::ConstSharedPtr & camera_data,
    const sensor_msgs::msg::Imu::ConstSharedPtr & imu_data);
  
  double getTerrainConfidence() const;

  static std::string terrainTypeToString(TerrainType type);

private:
  rclcpp::Node::SharedPtr node_;
  
  double slope_threshold_;
  double roughness_threshold_;
  
  TerrainType detectFromPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  TerrainType detectFromImage(const sensor_msgs::msg::Image::ConstSharedPtr & image);
  TerrainType detectFromIMU(const sensor_msgs::msg::Imu::ConstSharedPtr & imu_data);
  
  double terrain_confidence_;
  
  TerrainType current_terrain_type_;
};

}  // namespace terrain_aware_nav

#endif  // TERRAIN_AWARE_NAV__TERRAIN_DETECTOR_HPP_