// Copyright (c) 2025 Terrain Navigation Team
// Licensed under the Apache License, Version 2.0

#ifndef TERRAIN_AWARE_NAV__TERRAIN_CLASSIFIER_HPP_
#define TERRAIN_AWARE_NAV__TERRAIN_CLASSIFIER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "cv_bridge/cv_bridge.h"

#include "terrain_aware_nav/terrain_detection/terrain_detector.hpp"

namespace terrain_aware_nav
{

struct TerrainFeatures {
  double roughness;
  double slope;
  double hardness;
  double friction;
  std::vector<double> visual_features;
};

class TerrainClassifier
{
public:
  explicit TerrainClassifier(const rclcpp::Node::SharedPtr & node);
  virtual ~TerrainClassifier();

  // Initialize the classifier with parameters
  bool initialize();
  
  // Extract features from sensor data
  TerrainFeatures extractFeatures(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_data,
    const sensor_msgs::msg::Image::ConstSharedPtr & camera_data);
  
  // Classify terrain based on features
  TerrainType classifyTerrain(const TerrainFeatures & features);
  
  // Get confidence score of classification (0.0 to 1.0)
  double getClassificationConfidence() const;

private:
  // Node handle
  rclcpp::Node::SharedPtr node_;
  
  // Parameters
  std::string model_path_;
  bool use_ml_classification_;
  
  // Classification confidence
  double classification_confidence_;
  
  // Feature extraction methods
  double computeRoughness(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  double computeSlope(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  std::vector<double> extractVisualFeatures(const sensor_msgs::msg::Image::ConstSharedPtr & image);
  
  // Rule-based classification
  TerrainType applyClassificationRules(const TerrainFeatures & features);
  
  // ML-based classification (if enabled)
  TerrainType applyMLClassification(const TerrainFeatures & features);
};

}  // namespace terrain_aware_nav

#endif  // TERRAIN_AWARE_NAV__TERRAIN_CLASSIFIER_HPP_