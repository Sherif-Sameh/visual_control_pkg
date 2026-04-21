cmake_minimum_required(VERSION 3.8)
project(state_estimation_pkg)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++ Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(vc_core REQUIRED)
find_package(isaac_ros_apriltag REQUIRED)

# Include C++ "include" directory
include_directories(include)
include_directories(${vc_core_INCLUDE_DIRS})

# Create C++ library for composable nodes
add_library(apriltag_estimator_component SHARED src/apriltag_estimator.cpp)
ament_target_dependencies(apriltag_estimator_component 
  rclcpp
  rcl_interfaces
  rclcpp_components
  geometry_msgs
  sensor_msgs
  tf2_eigen
  tf2_ros
  isaac_ros_apriltag
)
rclcpp_components_register_node(
    apriltag_estimator_component
    PLUGIN "ApriltagEstimator"
    EXECUTABLE apriltag_estimator
)

add_library(pose_estimator_component SHARED src/pose_estimator.cpp)
ament_target_dependencies(pose_estimator_component 
  rclcpp
  rcl_interfaces
  rclcpp_components
  geometry_msgs
  sensor_msgs
  tf2_eigen
  tf2_ros
  isaac_ros_apriltag
)
rclcpp_components_register_node(
    pose_estimator_component
    PLUGIN "PoseEstimator"
    EXECUTABLE pose_estimator
)

# Install C++ composable nodes
ament_export_targets(export_apriltag_estimator_component)
install(TARGETS apriltag_estimator_component
        EXPORT export_apriltag_estimator_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

ament_export_targets(export_pose_estimator_component)
install(TARGETS pose_estimator_component
        EXPORT export_pose_estimator_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

# Install launch and config directories
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

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

ament_package()
