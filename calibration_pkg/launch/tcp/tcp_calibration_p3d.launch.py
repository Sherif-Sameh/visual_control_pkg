import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # TCP calibration arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_gt",
            default_value="[0.0]",
            description="Ground truth pose of TCP wrt camera if available."
            " To be used, the 7 pose parameters must be given in the order [tx, ty, tz, qw, qx, qy, qz]."
            " Default is [0.0], which is internally ignored.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "img_center",
            default_value="[240, 320]",
            description="Image center (i, j) for square crop to use for differentiable rendering."
            " Default is [240, 320].",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "image_topic_name",
            default_value="/image",
            description="Input image (sensor_msgs/Image) topic name. Default is /image.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "depth_topic_name",
            default_value="/depth",
            description="Input depth (sensor_msgs/Image) topic name. Default is /depth.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_info_topic_name",
            default_value="/camera_info",
            description="Camera info (sensor_msgs/CameraInfo) topic name. Default is /camera_info.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/tcp_calibration_p3d/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /tcp_calibration_p3d/restart.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    pose_gt = LaunchConfiguration("pose_gt")
    img_center = LaunchConfiguration("img_center")

    image_topic_name = LaunchConfiguration("image_topic_name")
    depth_topic_name = LaunchConfiguration("depth_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("calibration_pkg")
    config_path = os.path.join(pkg_share, "config", "tcp_calibration_p3d.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    tcp_calibration_p3d_node = Node(
        package="calibration_pkg",
        executable="tcp_calibration_p3d.py",
        output="screen",
        parameters=[{"pose_gt": pose_gt, "img_center": img_center, **config["calibration"]}],
        remappings=[
            ("/image", image_topic_name),
            ("/depth", depth_topic_name),
            ("/camera_info", camera_info_topic_name),
            ("/tcp_calibration_p3d/restart", restart_topic_name),
        ],
    )

    nodes_to_start = [tcp_calibration_p3d_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
