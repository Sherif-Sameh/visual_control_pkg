import os
from typing import Any

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
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
    declared_arguments.append(
        DeclareLaunchArgument(
            "modalities",
            default_value="['depth']",
            description="Rendering modalities to use for pose optimization. Options include"
            " 'silhouette' and 'depth' only. Default is ['depth'].",
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


def launch_setup(context: LaunchContext) -> list[Node]:
    # Initialize Arguments
    pose_gt = LaunchConfiguration("pose_gt")
    img_center = LaunchConfiguration("img_center")
    modalities = LaunchConfiguration("modalities")

    image_topic_name = LaunchConfiguration("image_topic_name")
    depth_topic_name = LaunchConfiguration("depth_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load parameters with applied overrides from toml
    pkg_share = get_package_share_directory("calibration_pkg")
    config_path = os.path.join(pkg_share, "config", "tcp_calibration_p3d.toml")
    config = toml.load(config_path)
    params = _load_params_with_overrides(config, modalities.perform(context))

    # Initialize TCP calibration node
    return [
        Node(
            package="calibration_pkg",
            executable="tcp_calibration_p3d.py",
            output="screen",
            parameters=[
                {
                    "pose_gt": pose_gt,
                    "img_center": img_center,
                    "dr.shader.modalities": modalities,
                    **params,
                }
            ],
            remappings=[
                ("/image", image_topic_name),
                ("/depth", depth_topic_name),
                ("/camera_info", camera_info_topic_name),
                ("/tcp_calibration_p3d/restart", restart_topic_name),
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Add opaque functions
    # node has to be init from opaque function to retrieve modality-specific overrides
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + opaque_functions)


##
# Private functions
##


def _load_params_with_overrides(config: dict[str, Any], modalities: str) -> dict[str, Any]:
    modalities = modalities.split(",")
    if len(modalities) == 2:
        overrides_key = "silhouette-depth"
    else:
        overrides_key = "silhouette" if "silhouette" in modalities else "depth"
    params = config["calibration"] | config[f"overrides-{overrides_key}"]
    return params
