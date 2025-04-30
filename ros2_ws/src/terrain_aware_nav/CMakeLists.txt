cmake_minimum_required(VERSION 3.8)
project(terrain_aware_nav)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)

# Include PCL headers
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Link PCL libraries for the targets
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add terrain detection library
add_library(terrain_detector
  src/terrain_detector.cpp
)
target_include_directories(terrain_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(terrain_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  PCL
  pcl_conversions
  OpenCV
)

# Add terrain classification library
add_library(terrain_classifier
  src/terrain_classifier.cpp
)
target_include_directories(terrain_classifier PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(terrain_classifier
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  PCL
  pcl_conversions
  OpenCV
)

# Add adaptive controller library
add_library(adaptive_controller
  src/adaptive_controller.cpp
)
target_include_directories(adaptive_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(adaptive_controller
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
)
target_link_libraries(adaptive_controller
  terrain_detector
)

# Add the executable for the main navigation node
add_executable(terrain_navigation_node
  src/terrain_navigation_node.cpp
)
target_include_directories(terrain_navigation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(terrain_navigation_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  PCL
  pcl_conversions
  OpenCV
)
target_link_libraries(terrain_navigation_node
  terrain_detector
  terrain_classifier
  adaptive_controller
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install libraries
install(
  TARGETS terrain_detector terrain_classifier adaptive_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executables
install(
  TARGETS terrain_navigation_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(
  DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(terrain_detector terrain_classifier adaptive_controller)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  PCL
  pcl_conversions
  OpenCV
)

ament_package()
