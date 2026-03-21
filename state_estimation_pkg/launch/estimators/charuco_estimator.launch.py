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

    # ChArUco estimator arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_size",
            default_value="0.3",
            description="Board size in meters (square board). Default value is 0.3.",
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
            description="ChArUco board detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    return declared_arguments


def get_composable_node(**kwargs) -> ComposableNode:
    # Initialize argument defaults for missing arguments
    defaults = {a.name: a.default_value for a in declare_arguments()}

    # Load configuration from toml
    pkg_share = get_package_share_directory("state_estimation_pkg")
    config_path = os.path.join(pkg_share, "config", "charuco_estimator.toml")
    config = toml.load(config_path)

    # Initialize composable node
    get_arg = lambda name: kwargs.get(name, defaults[name])  # noqa: E731
    return ComposableNode(
        package="state_estimation_pkg",
        plugin="ApriltagEstimator",
        name="charuco_estimator",
        parameters=[{"tag.size": get_arg("board_size"), **config["estimator"]}],
        remappings=[
            ("/apriltag_estimator/detections_filtered", "/charuco_estimator/detections_filtered"),
            ("/camera_info", get_arg("camera_info_topic_name")),
            ("/camera_twist", get_arg("camera_twist_topic_name")),
            ("/detections", get_arg("detections_topic_name")),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    board_size = LaunchConfiguration("board_size")

    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("state_estimation_pkg")
    config_path = os.path.join(pkg_share, "config", "charuco_estimator.toml")
    config = toml.load(config_path)

    # Initialize composable node
    charuco_estimator_node = ComposableNode(
        package="state_estimation_pkg",
        plugin="ApriltagEstimator",
        name="charuco_estimator",
        parameters=[{"tag.size": board_size, **config["estimator"]}],
        remappings=[
            ("/camera_info", camera_info_topic_name),
            ("/camera_twist", camera_twist_topic_name),
            ("/detections", detections_topic_name),
        ],
    )

    # Initialize standalone composable node container
    charuco_estimator_container = ComposableNodeContainer(
        package="rclcpp_components",
        name="charuco_estimator_container",
        namespace="",
        executable="component_container",
        composable_node_descriptions=[charuco_estimator_node],
        output="screen",
    )

    nodes_to_start = [charuco_estimator_container]
    return LaunchDescription(declared_arguments + nodes_to_start)
