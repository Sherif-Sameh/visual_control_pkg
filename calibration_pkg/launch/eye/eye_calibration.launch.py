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

    # Eye calibration arguments
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
            "marker_id",
            default_value="0",
            description="ID of the reference marker. Default value is 0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ref_pose",
            default_value="[0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
            description="Estimate of the target eye pose wrt to the reference marker."
            " Default value is [0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "model",
            default_value="mipmap",
            description="Eye pose + texture model type to use. Default value is mipmap.",
            choices=["simple", "mipmap", "hashenc"],
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
            "restart_topic_name",
            default_value="/eye_calibration/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /eye_calibration/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[Node]:
    # Initialize Arguments
    cam_frame = LaunchConfiguration("cam_frame")
    marker_frame = LaunchConfiguration("marker_frame")
    eye_gt_frame = LaunchConfiguration("eye_gt_frame")
    marker_id = LaunchConfiguration("marker_id")
    ref_pose = LaunchConfiguration("ref_pose")
    model = LaunchConfiguration("model")
    output_path = PathJoinSubstitution(
        [FindPackageShare("calibration_pkg"), "../../../../logs/output"]
    )
    mesh_path = PathJoinSubstitution(
        [FindPackageShare("calibration_pkg"), "../../../../logs/eye/eye.obj"]
    )

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load parameters with applied overrides from toml
    pkg_share = get_package_share_directory("calibration_pkg")
    config_path = os.path.join(pkg_share, "config", "eye_calibration.toml")
    config = toml.load(config_path)
    params = _load_params_with_overrides(config, model.perform(context))

    # Initialize eye calibration node
    return [
        Node(
            package="calibration_pkg",
            executable="eye_calibration.py",
            output="screen",
            parameters=[
                {
                    "output_path": output_path,
                    "frame.cam": cam_frame,
                    "frame.marker": marker_frame,
                    "frame.eye_gt": eye_gt_frame,
                    "ref.marker_id": marker_id,
                    "ref.pose": ref_pose,
                    "dr.mesh.path": mesh_path,
                    "dr.model.type": model,
                    **params,
                }
            ],
            remappings=[
                ("/image", image_topic_name),
                ("/camera_info", camera_info_topic_name),
                ("/eye_calibration/restart", restart_topic_name),
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


def _load_params_with_overrides(config: dict[str, Any], model: str) -> dict[str, Any]:
    overrides = config[f"overrides-{model}"]
    params = config["calibration"] | overrides
    return params
