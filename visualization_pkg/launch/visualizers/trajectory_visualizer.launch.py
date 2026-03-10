import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Trajectory visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "target_frames",
            default_value="['base_link']",
            description="Names of target frames for each tracked frame. Default is ['base_link'].",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "source_frames",
            default_value="['ee_link']",
            description="Names of source frames for each tracked frame. Default is ['ee_link'].",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/trajectory_visualizer/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /trajectory_visualizer/restart",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    target_frames = LaunchConfiguration("target_frames")
    source_frames = LaunchConfiguration("source_frames")

    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("visualization_pkg")
    config_path = os.path.join(pkg_share, "config", "trajectory_visualizer.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    trajectory_visualizer_node = Node(
        package="visualization_pkg",
        executable="trajectory_visualizer.py",
        output="screen",
        parameters=[
            {"frame.target": target_frames, "frame.source": source_frames, **config["visualizer"]}
        ],
        remappings=[("/trajectory_visualizer/restart", restart_topic_name)],
    )

    nodes_to_start = [trajectory_visualizer_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
