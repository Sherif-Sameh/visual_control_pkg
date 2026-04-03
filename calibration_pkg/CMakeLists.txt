cmake_minimum_required(VERSION 3.8)
project(calibration_pkg)

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
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(trajectory_msgs REQUIRED)

find_package(vc_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(isaac_ros_apriltag REQUIRED)

# Include C++ "include" directory
include_directories(include)
include_directories(${vc_core_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# Create C++ executables
add_executable(handeye_calibration src/handeye_calibration.cpp)
ament_target_dependencies(handeye_calibration 
  rclcpp
  rcl_interfaces
  std_msgs
  geometry_msgs
  tf2_eigen
  tf2_ros
  trajectory_msgs
  vc_core
  Eigen3
  isaac_ros_apriltag
)

add_executable(handeye_evaluation src/handeye_evaluation.cpp)
ament_target_dependencies(handeye_evaluation 
  rclcpp
  geometry_msgs
  tf2_eigen
  tf2_ros
  vc_core
  Eigen3
  isaac_ros_apriltag
)

# Install C++ executables
install(TARGETS
  handeye_calibration
  handeye_evaluation
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/tcp_calibration_p3d.py
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
