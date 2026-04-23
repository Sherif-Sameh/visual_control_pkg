import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Plan visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ref_frame",
            default_value="reference",
            description="Names of reference frame for the planned trajectory. Default is reference.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "planned_trajectory_topic_name",
            default_value="/planned_trajectory",
            description="Planned trajectory (trajectory_msgs/MultiDOFJointTrajectory) topic name."
            " Default is /planned_trajectory.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    ref_frame = LaunchConfiguration("ref_frame")

    planned_trajectory_topic_name = LaunchConfiguration("planned_trajectory_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("visualization_pkg")
    config_path = os.path.join(pkg_share, "config", "plan_visualizer.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    plan_visualizer_node = Node(
        package="visualization_pkg",
        executable="plan_visualizer.py",
        output="screen",
        parameters=[{"frame.ref": ref_frame, **config["visualizer"]}],
        remappings=[("/planned_trajectory", planned_trajectory_topic_name)],
    )

    nodes_to_start = [plan_visualizer_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
