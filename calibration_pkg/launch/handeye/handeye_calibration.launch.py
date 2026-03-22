import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Hand-eye calibration arguments
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
    declared_arguments.append(
        DeclareLaunchArgument(
            "cam_frame",
            default_value="camera_color_optical_frame",
            description="Name of the camera frame of the RGB images."
            " Default value is camera_color_optical_frame.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_gt",
            default_value="[0.0]",
            description="Ground truth pose of camera wrt robot end-effector link if available."
            " To be used, the 7 pose parameters must be given in the order [tx, ty, tz, qw, qx, qy, qz]."
            " Default value is [0.0], which is internally ignored.",
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
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/handeye_calibration/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /handeye_calibration/restart.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")
    cam_frame = LaunchConfiguration("cam_frame")
    calib_path = PathJoinSubstitution(
        [
            FindPackageShare("calibration_pkg"),
            "../../../../src/visual_control_pkg/calibration_pkg/config/handeye_params.toml",
        ]
    )
    pose_gt = LaunchConfiguration("pose_gt")

    detections_topic_name = LaunchConfiguration("detections_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("calibration_pkg")
    config_path = os.path.join(pkg_share, "config", "handeye_calibration.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    handeye_calibration_node = Node(
        package="calibration_pkg",
        executable="handeye_calibration",
        output="screen",
        parameters=[
            {
                "frame.base_frame": base_frame,
                "frame.ee_frame": ee_frame,
                "frame.cam_frame": cam_frame,
                "calib.path": calib_path,
                "calib.pose_gt": pose_gt,
                **config["calibration"],
            }
        ],
        remappings=[
            ("/detections", detections_topic_name),
            ("/handeye_calibration/restart", restart_topic_name),
        ],
    )

    nodes_to_start = [handeye_calibration_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
