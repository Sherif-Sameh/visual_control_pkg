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

    # OC planner arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "mode",
            default_value="default",
            description="OC Planner mode. Determines velocity and acceleration limits. "
            " Default valus is default.",
            choices=["default", "fast"],
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
            "tcp_frame",
            default_value="tcp",
            description="Name of the tcp frame of the robot. Default value is tcp.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_id",
            default_value="0",
            description="Tag ID to use for tracking. Default value is 0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_mk_tgt",
            default_value="[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
            description="Pose of target wrt the reference marker."
            " Default value is [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_reference_topic_name",
            default_value="/pose_reference",
            description="Reference pose (geometry_msgs/PoseStamped) topic name."
            " Default is /pose_reference",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_twist_topic_name",
            default_value="/camera_twist",
            description="Camera twist (geometry_msgs/TwistStamped) topic name."
            " Default is /camera_twist.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="AprilTag detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[Node]:
    # Initialize Arguments
    mode = LaunchConfiguration("mode")
    cam_frame = LaunchConfiguration("cam_frame")
    tcp_frame = LaunchConfiguration("tcp_frame")
    tag_id = LaunchConfiguration("tag_id")
    pose_mk_tgt = LaunchConfiguration("pose_mk_tgt")

    pose_reference_topic_name = LaunchConfiguration("pose_reference_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Load parameters with applied overrides from toml
    pkg_share = get_package_share_directory("control_pkg")
    config_path = os.path.join(pkg_share, "config", "oc_planner.toml")
    config = toml.load(config_path)
    params = _load_params_with_overrides(config, mode.perform(context))

    return [
        Node(
            package="control_pkg",
            executable="oc_planner.py",
            output="screen",
            parameters=[
                {
                    "frame.cam": cam_frame,
                    "frame.tcp": tcp_frame,
                    "pose.mk_tgt": pose_mk_tgt,
                    "tag.tag_id": tag_id,
                    **params,
                }
            ],
            remappings=[
                ("/pose_reference", pose_reference_topic_name),
                ("/camera_twist", camera_twist_topic_name),
                ("/detections", detections_topic_name),
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Add opaque functions
    # node has to be init from opaque function to retrieve mode-specific overrides
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + opaque_functions)


##
# Private functions
##


def _load_params_with_overrides(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "default":
        return config["planner"]
    overrides = config[f"overrides-{mode}"]
    params = config["planner"] | overrides
    return params
