import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Apriltag estimator arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_size",
            default_value="0.08",
            description="Tag size in meters of tracked tags. Default value is 0.08.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_info_topic_name",
            default_value="/camera_info",
            description="Camera info (sensor_msgs/CameraInfo) topic name.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_twist_topic_name",
            default_value="/camera_twist",
            description="Camera twist (geometry_msgs/TwistStamped) topic name.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="AprilTag raw detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    return declared_arguments


def get_composable_node(**kwargs) -> ComposableNode:
    # Initialize Arguments
    tag_size = LaunchConfiguration("tag_size")

    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("state_estimation_pkg")
    config_path = os.path.join(pkg_share, "config", "apriltag_estimator.toml")
    config = toml.load(config_path)

    # Initialize composable node
    return ComposableNode(
        package="state_estimation_pkg",
        plugin="ApriltagEstimator",
        parameters=[{"tag.size": kwargs.get("tag_size", tag_size), **config["estimator"]}],
        remappings=[
            ("/camera_info", kwargs.get("camera_info_topic_name", camera_info_topic_name)),
            ("/camera_twist", kwargs.get("camera_twist_topic_name", camera_twist_topic_name)),
            ("/detections", kwargs.get("detections_topic_name", detections_topic_name)),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize composable node
    apriltag_estimator_node = get_composable_node()

    # Initialize standalong composable node container
    apriltag_container = ComposableNodeContainer(
        package="rclcpp_components",
        name="apriltag_estimator_container",
        namespace="",
        executable="component_container",
        composable_node_descriptions=[apriltag_estimator_node],
        output="screen",
    )

    nodes_to_start = [apriltag_container]
    return LaunchDescription(declared_arguments + nodes_to_start)
