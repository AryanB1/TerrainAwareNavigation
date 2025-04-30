#include "terrain_aware_nav/terrain_classification/terrain_classifier.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

namespace terrain_aware_nav
{

TerrainClassifier::TerrainClassifier(const rclcpp::Node::SharedPtr & node)
: node_(node),
  classification_confidence_(0.0),
  use_ml_classification_(false),
  model_path_("")
{
}

TerrainClassifier::~TerrainClassifier()
{
}

bool TerrainClassifier::initialize()
{
  node_->declare_parameter("use_ml_classification", false);
  node_->declare_parameter("ml_model_path", "");
  
  use_ml_classification_ = node_->get_parameter("use_ml_classification").as_bool();
  model_path_ = node_->get_parameter("ml_model_path").as_string();
  
  RCLCPP_INFO(
    node_->get_logger(),
    "TerrainClassifier initialized with ML classification: %s",
    use_ml_classification_ ? "enabled" : "disabled");
    
  if (use_ml_classification_) {
    if (!loadTFLiteModel()) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Failed to load TFLite model. Falling back to rule-based classification.");
      use_ml_classification_ = false;
    } else {
      RCLCPP_INFO(
        node_->get_logger(),
        "TensorFlow Lite model loaded successfully from: %s",
        model_path_.c_str());
    }
  }

  return true;
}

bool TerrainClassifier::loadTFLiteModel()
{
  if (model_path_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No model path provided for TFLite model");
    return false;
  }

  try {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to load TFLite model from: %s", model_path_.c_str());
      return false;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    
    if (!interpreter_) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create TFLite interpreter");
      return false;
    }
    
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to allocate tensors in TFLite interpreter");
      return false;
    }
    
    RCLCPP_DEBUG(node_->get_logger(), "TFLite model loaded successfully");
    RCLCPP_DEBUG(node_->get_logger(), "Input tensor count: %d", interpreter_->inputs().size());
    RCLCPP_DEBUG(node_->get_logger(), "Output tensor count: %d", interpreter_->outputs().size());
    
    return true;
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Exception while loading TFLite model: %s", e.what());
    return false;
  }
}

TerrainFeatures TerrainClassifier::extractFeatures(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_data,
  const sensor_msgs::msg::Image::ConstSharedPtr & camera_data)
{
  TerrainFeatures features;
  features.roughness = 0.0;
  features.slope = 0.0;
  features.hardness = 0.5;
  features.friction = 0.5;
  
  if (lidar_data) {
    features.roughness = computeRoughness(lidar_data);
    features.slope = computeSlope(lidar_data);
  }
  
  if (camera_data) {
    features.visual_features = extractVisualFeatures(camera_data);
  }
  
  return features;
}

double TerrainClassifier::computeRoughness(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
{
  if (!cloud || cloud->data.empty()) {
    return 0.0;
  }
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud, *pcl_cloud);
  
  if (pcl_cloud->empty()) {
    return 0.0;
  }
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(pcl_cloud);
  voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f);
  voxel_filter.filter(*filtered_cloud);
  
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.setMaxIterations(100);
  
  seg.setInputCloud(filtered_cloud);
  seg.segment(*inliers, *coefficients);
  
  if (inliers->indices.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Could not estimate a planar model for the given point cloud");
    return 0.5;
  }
  
  double total_distance = 0.0;
  double max_distance = 0.0;
  
  for (const auto & point : filtered_cloud->points) {
    double distance = std::abs(
      coefficients->values[0] * point.x +
      coefficients->values[1] * point.y +
      coefficients->values[2] * point.z +
      coefficients->values[3]) /
      std::sqrt(
        coefficients->values[0] * coefficients->values[0] +
        coefficients->values[1] * coefficients->values[1] +
        coefficients->values[2] * coefficients->values[2]);
    
    total_distance += distance;
    max_distance = std::max(max_distance, distance);
  }
  
  double avg_distance = total_distance / filtered_cloud->size();
  
  constexpr double max_expected_roughness = 0.1;
  double roughness = std::min(avg_distance / max_expected_roughness, 1.0);
  
  RCLCPP_DEBUG(
    node_->get_logger(),
    "Computed terrain roughness: %.3f (avg distance: %.3f, max: %.3f)",
    roughness, avg_distance, max_distance);
  
  return roughness;
}

double TerrainClassifier::computeSlope(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
{
  if (!cloud || cloud->data.empty()) {
    return 0.0;
  }
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud, *pcl_cloud);
  
  if (pcl_cloud->empty()) {
    return 0.0;
  }
  
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.setMaxIterations(100);
  
  seg.setInputCloud(pcl_cloud);
  seg.segment(*inliers, *coefficients);
  
  if (inliers->indices.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Could not estimate a planar model for the given point cloud");
    return 0.0;
  }
  
  double a = coefficients->values[0];
  double b = coefficients->values[1];
  double c = coefficients->values[2];
  
  double norm = std::sqrt(a * a + b * b + c * c);
  a /= norm;
  b /= norm;
  c /= norm;
  
  double angle_rad = std::acos(std::abs(c));
  
  constexpr double max_slope_rad = 0.785398;
  double slope = std::min(angle_rad / max_slope_rad, 1.0);
  
  RCLCPP_DEBUG(
    node_->get_logger(),
    "Computed terrain slope: %.3f (%.1f degrees)",
    slope, angle_rad * 180.0 / M_PI);
  
  return slope;
}

std::vector<double> TerrainClassifier::extractVisualFeatures(
  const sensor_msgs::msg::Image::ConstSharedPtr & image)
{
  std::vector<double> features;
  
  if (!image) {
    features.resize(5, 0.0);
    return features;
  }
  
  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image, "bgr8");
    cv::Mat img = cv_ptr->image;
    
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    
    cv::Mat hue_hist;
    int h_bins = 8;
    float h_range[] = {0, 180};
    const float * h_ranges = {h_range};
    cv::calcHist(&hsv_channels[0], 1, 0, cv::Mat(), hue_hist, 1, &h_bins, &h_ranges);
    cv::normalize(hue_hist, hue_hist, 0, 1, cv::NORM_MINMAX);
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    cv::Mat lbp = gray.clone();
    for (int i = 1; i < gray.rows - 1; i++) {
      for (int j = 1; i < gray.cols - 1; j++) {
        uchar center = gray.at<uchar>(i, j);
        uchar code = 0;
        if (gray.at<uchar>(i-1, j-1) > center) code |= 0x01;
        if (gray.at<uchar>(i-1, j  ) > center) code |= 0x02;
        if (gray.at<uchar>(i-1, j+1) > center) code |= 0x04;
        if (gray.at<uchar>(i  , j-1) > center) code |= 0x08;
        if (gray.at<uchar>(i  , j+1) > center) code |= 0x10;
        if (gray.at<uchar>(i+1, j-1) > center) code |= 0x20;
        if (gray.at<uchar>(i+1, j  ) > center) code |= 0x40;
        if (gray.at<uchar>(i+1, j+1) > center) code |= 0x80;
        lbp.at<uchar>(i, j) = code;
      }
    }
    
    cv::Mat lbp_hist;
    int l_bins = 16;
    int l_channels[] = {0};
    int l_histsize[] = {l_bins};
    float l_range[] = {0, 256};
    const float * l_ranges[] = {l_range};
    cv::calcHist(&lbp, 1, l_channels, cv::Mat(), lbp_hist, 1, l_histsize, l_ranges);
    cv::normalize(lbp_hist, lbp_hist, 0, 1, cv::NORM_MINMAX);
    
    cv::Scalar hsv_mean = cv::mean(hsv);
    features.push_back(hsv_mean[0] / 180.0);
    features.push_back(hsv_mean[1] / 255.0);
    features.push_back(hsv_mean[2] / 255.0);
    
    features.push_back(stddev[0] / 128.0);
    
    for (int i = 0; i < std::min(3, h_bins); i++) {
      features.push_back(hue_hist.at<float>(i));
    }
    
    for (int i = 0; i < std::min(3, l_bins); i++) {
      features.push_back(lbp_hist.at<float>(i * l_bins / 3));
    }
    
    int green_pixels = 0;
    int blue_pixels = 0;
    int yellow_pixels = 0;
    
    for (int i = 0; i < img.rows; i += 10) {
      for (int j = 0; j < img.cols; j += 10) {
        cv::Vec3b hsv_pixel = hsv.at<cv::Vec3b>(i, j);
        if (hsv_pixel[0] > 35 && hsv_pixel[0] < 85 && hsv_pixel[1] > 50) {
          green_pixels++;
        }
        else if (hsv_pixel[0] > 90 && hsv_pixel[0] < 130 && hsv_pixel[1] > 50) {
          blue_pixels++;
        }
        else if (hsv_pixel[0] > 15 && hsv_pixel[0] < 35 && hsv_pixel[1] > 50) {
          yellow_pixels++;
        }
      }
    }
    
    int total_sampled = img.rows * img.cols / 100;
    features.push_back(static_cast<double>(green_pixels) / total_sampled);
    features.push_back(static_cast<double>(blue_pixels) / total_sampled);
    features.push_back(static_cast<double>(yellow_pixels) / total_sampled);
    
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "CV bridge exception: %s", e.what());
    features.resize(15, 0.0);
  }
  
  return features;
}

TerrainType TerrainClassifier::classifyTerrain(const TerrainFeatures & features)
{
  if (use_ml_classification_) {
    TerrainType ml_prediction = applyMLClassification(features);
    if (ml_prediction != TerrainType::UNKNOWN) {
      return ml_prediction;
    }
  }
  
  return applyClassificationRules(features);
}

TerrainType TerrainClassifier::applyClassificationRules(const TerrainFeatures & features)
{
  if (features.slope > 0.7) {
    classification_confidence_ = 0.7 + 0.3 * features.slope;
    return TerrainType::STEEP;
  }
  
  if (features.roughness > 0.6) {
    classification_confidence_ = 0.6 + 0.4 * features.roughness;
    return TerrainType::ROUGH;
  }
  
  if (!features.visual_features.empty() && features.visual_features.size() >= 12) {
    double green_ratio = features.visual_features[12];
    if (green_ratio > 0.4) {
      classification_confidence_ = 0.6 + 0.4 * green_ratio;
      return TerrainType::SOFT;
    }
  }
  
  if (!features.visual_features.empty() && features.visual_features.size() >= 13) {
    double blue_ratio = features.visual_features[13];
    if (blue_ratio > 0.3) {
      classification_confidence_ = 0.6 + 0.4 * blue_ratio;
      return TerrainType::SLIPPERY;
    }
  }
  
  if (features.roughness < 0.3 && features.slope < 0.3) {
    classification_confidence_ = 0.8 - features.roughness - features.slope;
    return TerrainType::FLAT;
  }
  
  classification_confidence_ = 0.4;
  return TerrainType::UNKNOWN;
}

TerrainType TerrainClassifier::applyMLClassification(const TerrainFeatures & features)
{
  if (use_ml_classification_ && interpreter_) {
    std::vector<float> input_features;
    
    input_features.push_back(static_cast<float>(features.roughness));
    input_features.push_back(static_cast<float>(features.slope));
    
    for (const auto& feature : features.visual_features) {
      input_features.push_back(static_cast<float>(feature));
    }
    
    return runTFLiteInference(input_features);
  }
  
  try {
    if (features.visual_features.size() < 10) {
      RCLCPP_WARN(node_->get_logger(), "Not enough visual features for ML classification");
      return TerrainType::UNKNOWN;
    }
    
    double roughness = features.roughness;
    double slope = features.slope;
    double green_ratio = 0.0;
    double blue_ratio = 0.0;
    double yellow_ratio = 0.0;
    
    if (features.visual_features.size() >= 13) {
      green_ratio = features.visual_features[12];
      blue_ratio = features.visual_features[13];
      if (features.visual_features.size() >= 14) {
        yellow_ratio = features.visual_features[14];
      }
    }
    
    if (roughness > 0.8) {
      classification_confidence_ = 0.85;
      return TerrainType::OBSTACLE;
    }
    
    if (slope > 0.65) {
      classification_confidence_ = 0.75 + 0.2 * slope;
      return TerrainType::STEEP;
    }
    
    if (blue_ratio > 0.35 && roughness < 0.3) {
      classification_confidence_ = 0.7 + 0.3 * blue_ratio;
      return TerrainType::SLIPPERY;
    }
    
    if (green_ratio > 0.4) {
      classification_confidence_ = 0.75 + 0.25 * green_ratio;
      return TerrainType::SOFT;
    }
    
    if (yellow_ratio > 0.3) {
      if (roughness > 0.4) {
        classification_confidence_ = 0.65 + 0.2 * roughness;
        return TerrainType::ROUGH;
      } else {
        classification_confidence_ = 0.7;
        return TerrainType::SOFT;
      }
    }
    
    if (roughness > 0.5) {
      classification_confidence_ = 0.6 + 0.3 * roughness;
      return TerrainType::ROUGH;
    }
    
    if (roughness < 0.25 && slope < 0.2) {
      classification_confidence_ = 0.8;
      return TerrainType::FLAT;
    }
    
    classification_confidence_ = 0.5;
    return TerrainType::UNKNOWN;
    
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      node_->get_logger(), 
      "ML classification error: %s", e.what());
    return TerrainType::UNKNOWN;
  }
}

TerrainType TerrainClassifier::runTFLiteInference(const std::vector<float> & input_features)
{
  try {
    int input_tensor_idx = interpreter_->inputs()[0];
    TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_idx);
    
    TfLiteIntArray* input_dims = input_tensor->dims;
    int input_size = 1;
    for (int i = 0; i < input_dims->size; i++) {
      input_size *= input_dims->data[i];
    }
    
    if (input_size != static_cast<int>(input_features.size())) {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Input feature size mismatch: got %zu, expected %d",
        input_features.size(), input_size);
      return TerrainType::UNKNOWN;
    }
    
    float* input_data = interpreter_->typed_input_tensor<float>(0);
    for (size_t i = 0; i < input_features.size(); i++) {
      input_data[i] = input_features[i];
    }
    
    if (interpreter_->Invoke() != kTfLiteOk) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to invoke TFLite interpreter");
      return TerrainType::UNKNOWN;
    }
    
    int output_tensor_idx = interpreter_->outputs()[0];
    TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx);
    float* output_data = interpreter_->typed_output_tensor<float>(0);
    
    int num_classes = output_tensor->dims->data[output_tensor->dims->size - 1];
    int predicted_class = 0;
    float highest_prob = 0.0f;
    
    for (int i = 0; i < num_classes; i++) {
      if (output_data[i] > highest_prob) {
        highest_prob = output_data[i];
        predicted_class = i;
      }
    }
    
    TerrainType terrain_type = static_cast<TerrainType>(predicted_class);
    
    classification_confidence_ = static_cast<double>(highest_prob);
    
    RCLCPP_DEBUG(
      node_->get_logger(),
      "TFLite inference result: class %d (%s) with confidence %.2f",
      predicted_class,
      TerrainDetector::terrainTypeToString(terrain_type).c_str(),
      classification_confidence_);
    
    return terrain_type;
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Exception during TFLite inference: %s", e.what());
    return TerrainType::UNKNOWN;
  }
}

double TerrainClassifier::getClassificationConfidence() const
{
  return classification_confidence_;
}

}  // namespace terrain_aware_nav