cmake_minimum_required(VERSION 3.8)
project(vision_pkg)

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

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

find_package(vc_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(isaac_ros_apriltag REQUIRED)

# Include C++ "include" directory
include_directories(include)
include_directories(${vc_core_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# Create C++ library for composable nodes
add_library(charuco_detector_component SHARED src/charuco_detector.cpp)
ament_target_dependencies(charuco_detector_component 
  rclcpp
  rcl_interfaces
  geometry_msgs
  sensor_msgs
  tf2_ros
  tf2_eigen
  vc_core
  Eigen3
  OpenCV
  cv_bridge
  isaac_ros_apriltag
)
rclcpp_components_register_node(
    charuco_detector_component
    PLUGIN "CharucoDetector"
    EXECUTABLE charuco_detector
)

# Install C++ composable nodes
ament_export_targets(export_charuco_detector_component)
install(TARGETS charuco_detector_component
        EXPORT export_charuco_detector_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

# Install Python executables
install(PROGRAMS
  scripts/eye_detector.py
  DESTINATION lib/${PROJECT_NAME}
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
