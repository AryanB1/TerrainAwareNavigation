#include "terrain_aware_nav/terrain_detection/terrain_detector.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>

namespace terrain_aware_nav
{

TerrainDetector::TerrainDetector(const rclcpp::Node::SharedPtr & node)
: node_(node),
  terrain_confidence_(0.0),
  current_terrain_type_(TerrainType::UNKNOWN),
  slope_threshold_(0.5),
  roughness_threshold_(0.2)
{
}

TerrainDetector::~TerrainDetector()
{
}

bool TerrainDetector::initialize()
{
  // Get parameters from ROS param server
  node_->declare_parameter("slope_threshold", 0.5);
  node_->declare_parameter("roughness_threshold", 0.2);
  
  slope_threshold_ = node_->get_parameter("slope_threshold").as_double();
  roughness_threshold_ = node_->get_parameter("roughness_threshold").as_double();
  
  RCLCPP_INFO(
    node_->get_logger(),
    "TerrainDetector initialized with slope_threshold: %f, roughness_threshold: %f",
    slope_threshold_, roughness_threshold_);

  return true;
}

TerrainType TerrainDetector::detectTerrain(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_data,
  const sensor_msgs::msg::Image::ConstSharedPtr & camera_data,
  const sensor_msgs::msg::Imu::ConstSharedPtr & imu_data)
{
  // Confidence levels for different detection methods
  double lidar_confidence = 0.6;
  double camera_confidence = 0.3;
  double imu_confidence = 0.1;
  
  // Get terrain classifications from different sources
  TerrainType lidar_terrain = detectFromPointCloud(lidar_data);
  TerrainType camera_terrain = detectFromImage(camera_data);
  TerrainType imu_terrain = detectFromIMU(imu_data);
  
  // Simple confidence-weighted voting
  std::unordered_map<TerrainType, double> terrain_votes;
  
  terrain_votes[lidar_terrain] += lidar_confidence;
  terrain_votes[camera_terrain] += camera_confidence;
  terrain_votes[imu_terrain] += imu_confidence;
  
  // Find terrain type with highest confidence
  TerrainType highest_vote_terrain = TerrainType::UNKNOWN;
  double highest_vote = 0.0;
  
  for (const auto & vote : terrain_votes) {
    if (vote.second > highest_vote) {
      highest_vote = vote.second;
      highest_vote_terrain = vote.first;
    }
  }
  
  // Update confidence and current terrain
  terrain_confidence_ = highest_vote;
  current_terrain_type_ = highest_vote_terrain;
  
  return current_terrain_type_;
}

TerrainType TerrainDetector::detectFromPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
{
  if (!cloud) {
    return TerrainType::UNKNOWN;
  }
  
  // Convert PointCloud2 to PCL format
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud, *pcl_cloud);
  
  // Downsample cloud for efficiency
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(pcl_cloud);
  voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f);
  voxel_filter.filter(*filtered_cloud);
  
  // Estimate normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  
  normal_estimator.setInputCloud(filtered_cloud);
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setKSearch(10);  // Use 10 neighbors
  normal_estimator.compute(*normals);
  
  // Calculate terrain characteristics
  double avg_slope = 0.0;
  double avg_roughness = 0.0;
  int valid_points = 0;
  
  for (size_t i = 0; i < normals->size(); ++i) {
    // Calculate slope as angle from vertical (dot product with [0,0,1])
    double slope = std::acos(std::abs(normals->points[i].normal_z));
    avg_slope += slope;
    
    // Roughness estimation using normal variation
    // (simplified - in practice would use more sophisticated measures)
    if (i > 0) {
      double roughness = std::abs(normals->points[i].normal_z - normals->points[i - 1].normal_z);
      avg_roughness += roughness;
    }
    
    valid_points++;
  }
  
  // Average calculations
  if (valid_points > 0) {
    avg_slope /= valid_points;
    avg_roughness /= (valid_points - 1);
  }
  
  // Terrain classification based on slope and roughness
  if (avg_roughness > roughness_threshold_ && avg_slope > slope_threshold_) {
    return TerrainType::ROUGH;
  } else if (avg_slope > slope_threshold_ * 1.5) {
    return TerrainType::STEEP;
  } else if (avg_roughness <= roughness_threshold_ * 0.3) {
    return TerrainType::FLAT;
  } else {
    return TerrainType::UNKNOWN;
  }
}

TerrainType TerrainDetector::detectFromImage(const sensor_msgs::msg::Image::ConstSharedPtr & image)
{
  if (!image) {
    return TerrainType::UNKNOWN;
  }
  
  try {
    // Convert ROS image to OpenCV format
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image, "bgr8");
    cv::Mat img = cv_ptr->image;
    
    // Simple color-based terrain detection (this is very basic - would use ML in production)
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    
    // Split the HSV channels
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    
    // Calculate metrics from the image
    cv::Scalar mean_hsv = cv::mean(hsv);
    cv::Scalar std_dev;
    cv::meanStdDev(hsv_channels[2], mean_hsv, std_dev);  // Value channel variance for texture
    
    // Very simple rules for terrain types based on colors and texture
    double texture = std_dev[0];
    
    // Green indicates grass/vegetation
    int green_pixels = 0;
    for (int i = 0; i < img.rows; i++) {
      for (int j = 0; j < img.cols; j++) {
        cv::Vec3b hsv_pixel = hsv.at<cv::Vec3b>(i, j);
        if (hsv_pixel[0] > 35 && hsv_pixel[0] < 85 && hsv_pixel[1] > 50) {
          green_pixels++;
        }
      }
    }
    
    double green_ratio = static_cast<double>(green_pixels) / (img.rows * img.cols);
    
    // Simple terrain classification based on color and texture
    if (green_ratio > 0.5) {
      return TerrainType::SOFT;  // Likely grass or vegetation
    } else if (texture > 50.0) {
      return TerrainType::ROUGH;
    } else {
      return TerrainType::FLAT;
    }
    
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "CV bridge exception: %s", e.what());
    return TerrainType::UNKNOWN;
  }
}

TerrainType TerrainDetector::detectFromIMU(const sensor_msgs::msg::Imu::ConstSharedPtr & imu_data)
{
  if (!imu_data) {
    return TerrainType::UNKNOWN;
  }
  
  // Extract acceleration data
  double accel_x = imu_data->linear_acceleration.x;
  double accel_y = imu_data->linear_acceleration.y;
  double accel_z = imu_data->linear_acceleration.z;
  
  // Calculate vibration (higher on rough terrain)
  static std::vector<double> accel_history_x;
  static std::vector<double> accel_history_y;
  static std::vector<double> accel_history_z;
  
  // Keep a history window of 50 readings
  const size_t history_size = 50;
  
  accel_history_x.push_back(accel_x);
  accel_history_y.push_back(accel_y);
  accel_history_z.push_back(accel_z);
  
  if (accel_history_x.size() > history_size) {
    accel_history_x.erase(accel_history_x.begin());
    accel_history_y.erase(accel_history_y.begin());
    accel_history_z.erase(accel_history_z.begin());
  }
  
  if (accel_history_x.size() < 10) {
    return TerrainType::UNKNOWN;  // Not enough data
  }
  
  // Calculate variance in each axis
  double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;
  double var_x = 0.0, var_y = 0.0, var_z = 0.0;
  
  // Calculate mean
  for (size_t i = 0; i < accel_history_x.size(); i++) {
    mean_x += accel_history_x[i];
    mean_y += accel_history_y[i];
    mean_z += accel_history_z[i];
  }
  
  mean_x /= accel_history_x.size();
  mean_y /= accel_history_y.size();
  mean_z /= accel_history_z.size();
  
  // Calculate variance
  for (size_t i = 0; i < accel_history_x.size(); i++) {
    var_x += (accel_history_x[i] - mean_x) * (accel_history_x[i] - mean_x);
    var_y += (accel_history_y[i] - mean_y) * (accel_history_y[i] - mean_y);
    var_z += (accel_history_z[i] - mean_z) * (accel_history_z[i] - mean_z);
  }
  
  var_x /= accel_history_x.size();
  var_y /= accel_history_y.size();
  var_z /= accel_history_z.size();
  
  // Total vibration metric
  double vibration = var_x + var_y + var_z;
  
  // Classify terrain based on vibration and gravity direction
  if (vibration > 0.5) {
    return TerrainType::ROUGH;
  } else if (std::abs(accel_z - 9.81) > 1.0) {
    return TerrainType::STEEP;
  } else if (vibration < 0.1) {
    return TerrainType::SLIPPERY;  // Low friction surfaces often have less vibration
  } else {
    return TerrainType::FLAT;
  }
}

double TerrainDetector::getTerrainConfidence() const
{
  return terrain_confidence_;
}

std::string TerrainDetector::terrainTypeToString(TerrainType type)
{
  switch (type) {
    case TerrainType::FLAT:
      return "FLAT";
    case TerrainType::ROUGH:
      return "ROUGH";
    case TerrainType::STEEP:
      return "STEEP";
    case TerrainType::SLIPPERY:
      return "SLIPPERY";
    case TerrainType::SOFT:
      return "SOFT";
    case TerrainType::OBSTACLE:
      return "OBSTACLE";
    case TerrainType::UNKNOWN:
    default:
      return "UNKNOWN";
  }
}

}  // namespace terrain_aware_nav