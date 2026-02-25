cmake_minimum_required(VERSION 3.8)
project(visual_control_pkg)

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
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED) 
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

find_package(isaac_ros_apriltag)
find_package(Eigen3 REQUIRED)
find_package(VISP COMPONENTS core robot visual_features vs REQUIRED)
find_package(orocos_kdl_vendor REQUIRED)
find_package(kdl_parser REQUIRED)

# Include C++ "include" directory
include_directories(include/visual_control_pkg)
include_directories(${VISP_INCLUDE_DIRS})

## Declare a C++ library
add_library(vc_lib STATIC 
  lib/kdlIkSolverVel_wdls.cpp 
  lib/vpRobotRos.cpp
)

ament_target_dependencies(vc_lib 
  VISP 
  orocos_kdl_vendor 
  kdl_parser
)

# Create C++ executables
add_executable(pbvs_controller src/pbvs_controller.cpp)
target_link_libraries(pbvs_controller vc_lib)
ament_target_dependencies(pbvs_controller 
  rclcpp
  geometry_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  trajectory_msgs
  isaac_ros_apriltag
  VISP
)

add_executable(ibvs_controller src/ibvs_controller.cpp)
target_link_libraries(ibvs_controller vc_lib)
ament_target_dependencies(ibvs_controller 
  rclcpp
  geometry_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  trajectory_msgs
  isaac_ros_apriltag
  VISP
)

# Install C++ executables
install(TARGETS
  pbvs_controller
  ibvs_controller
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/ros_logger.py
  scripts/visualizers/detection_visualizer.py
  scripts/visualizers/trajectory_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and rviz directories
install(DIRECTORY launch rviz
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
