import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "image_topic_name",
            default_value="/image",
            description="Input image (sensor_msgs/Image) topic name to use for visualizer.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="Detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray) topic"
            " name to use for visualizer.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    image_topic_name = LaunchConfiguration("image_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("visualization_pkg")
    config_path = os.path.join(pkg_share, "config", "detection_visualizer.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    detection_visualizer_node = Node(
        package="visualization_pkg",
        executable="detection_visualizer.py",
        output="screen",
        parameters=[{**config["visualizer"]}],
        remappings=[("/image", image_topic_name), ("/detections", detections_topic_name)],
    )

    nodes_to_start = [detection_visualizer_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
