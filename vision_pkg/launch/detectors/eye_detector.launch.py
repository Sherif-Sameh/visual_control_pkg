import os
from typing import Any

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Eye detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "marker_frame",
            default_value="tag36h11:0",
            description="Name of the reference marker frame. Default value is tag36h11:0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "eye_gt_frame",
            default_value="eye_left",
            description="Name of the ground truth target eye frame. Default value is eye_left.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ref_pose",
            default_value="[0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
            description="Estimate of the eye pose wrt to the reference marker."
            " Default value is [0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "backend",
            default_value="cuda",
            description="Kaolin differentiable rendering backend. Default value is cuda.",
            choices=["cuda", "nvdiffrast"],
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
            "camera_info_topic_name",
            default_value="/camera_info",
            description="Camera info (sensor_msgs/CameraInfo) topic name. Default is /camera_info.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_topic_name",
            default_value="/eye_detector/pose_pred",
            description="Pose predictions (geometry_msgs/PoseStamped) topic name."
            " Default is /eye_detector/pose_pred.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/eye_detector/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /eye_detector/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[Node]:
    # Initialize Arguments
    marker_frame = LaunchConfiguration("marker_frame")
    eye_gt_frame = LaunchConfiguration("eye_gt_frame")
    ref_pose = LaunchConfiguration("ref_pose")
    backend = LaunchConfiguration("backend")
    mesh_path = PathJoinSubstitution(
        [FindPackageShare("vision_pkg"), "../../../../logs/eye/eye.obj"]
    )
    texture_path = PathJoinSubstitution(
        [FindPackageShare("vision_pkg"), "../../../../logs/output/texture.png"]
    )

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    pose_topic_name = LaunchConfiguration("pose_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load parameters with applied overrides from toml
    pkg_share = get_package_share_directory("vision_pkg")
    config_path = os.path.join(pkg_share, "config", "eye_detector.toml")
    config = toml.load(config_path)
    params = _load_params_with_overrides(config, backend.perform(context))

    # Initialize eye detector node
    return [
        Node(
            package="vision_pkg",
            executable="eye_detector.py",
            output="screen",
            parameters=[
                {
                    "frame.marker": marker_frame,
                    "frame.eye_gt": eye_gt_frame,
                    "ref.pose": ref_pose,
                    "dr.mesh.path": mesh_path,
                    "dr.mesh.texture": texture_path,
                    "dr.raster.backend": backend,
                    **params,
                }
            ],
            remappings=[
                ("/image", image_topic_name),
                ("/camera_info", camera_info_topic_name),
                ("/eye_detector/pose_pred", pose_topic_name),
                ("/eye_detector/restart", restart_topic_name),
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Add opaque functions
    # node has to be init from opaque function to retrieve model-specific overrides
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + opaque_functions)


##
# Private functions
##


def _load_params_with_overrides(config: dict[str, Any], backend: str) -> dict[str, Any]:
    overrides = config[f"overrides-{backend}"]
    params = config["detector"] | overrides
    return params
