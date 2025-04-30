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

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

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

  bool initialize();
  
  TerrainFeatures extractFeatures(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_data,
    const sensor_msgs::msg::Image::ConstSharedPtr & camera_data);
  
  TerrainType classifyTerrain(const TerrainFeatures & features);
  
  double getClassificationConfidence() const;

private:
  rclcpp::Node::SharedPtr node_;
  
  std::string model_path_;
  bool use_ml_classification_;
  
  double classification_confidence_;
  
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  
  bool loadTFLiteModel();
  
  double computeRoughness(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  double computeSlope(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  std::vector<double> extractVisualFeatures(const sensor_msgs::msg::Image::ConstSharedPtr & image);
  
  TerrainType applyClassificationRules(const TerrainFeatures & features);
  TerrainType applyMLClassification(const TerrainFeatures & features);
  
  TerrainType runTFLiteInference(const std::vector<float> & input_features);
};

}  // namespace terrain_aware_nav

#endif  // TERRAIN_AWARE_NAV__TERRAIN_CLASSIFIER_HPP_