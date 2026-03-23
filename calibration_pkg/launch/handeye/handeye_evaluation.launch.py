import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Hand-eye calibration evaluation arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Name of the base frame of the robot. Default value is base_link.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ee_frame",
            default_value="ee_link",
            description="Name of the end-effector frame of the robot. Default value is ee_link.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="AprilTag detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")

    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Load calibration parameters from toml
    pkg_share = get_package_share_directory("calibration_pkg")
    calib_path = os.path.join(
        pkg_share, "../../../../src/visual_control_pkg/calibration_pkg/config/handeye_params.toml"
    )
    calib_params = toml.load(calib_path)["pose"]
    pose_est = calib_params["translation"] + calib_params["rotation"]

    # Load configuration from toml
    config_path = os.path.join(pkg_share, "config", "handeye_evaluation.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    handeye_evaluation_node = Node(
        package="calibration_pkg",
        executable="handeye_evaluation",
        output="screen",
        parameters=[
            {
                "frame.base_frame": base_frame,
                "frame.ee_frame": ee_frame,
                "calib.pose_est": pose_est,
                **config["calibration"],
            }
        ],
        remappings=[("/detections", detections_topic_name)],
    )

    nodes_to_start = [handeye_evaluation_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
