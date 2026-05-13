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

    # Pose estimator arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_frame",
            default_value="pose",
            description="Name of the estimated pose frame. Default value is pose.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_twist_topic_name",
            default_value="/camera_twist",
            description="Camera twist (geometry_msgs/TwistStamped) topic name.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_topic_name",
            default_value="/pose",
            description="Pose raw measurements (isaac_ros_apriltag_interfaces/AprilTagDetectionArray) topic name."
            " Default is /pose.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/pose_estimator/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /pose_estimator/restart.",
        )
    )
    return declared_arguments


def get_composable_node(**kwargs) -> ComposableNode:
    # Initialize argument defaults for missing arguments
    defaults = {a.name: a.default_value for a in declare_arguments()}

    # Load configuration from toml
    pkg_share = get_package_share_directory("state_estimation_pkg")
    config_path = os.path.join(pkg_share, "config", "pose_estimator.toml")
    config = toml.load(config_path)

    # Initialize composable node
    get_arg = lambda name: kwargs.get(name, defaults[name])  # noqa: E731
    return ComposableNode(
        package="state_estimation_pkg",
        plugin="PoseEstimator",
        parameters=[{"pose.frame": get_arg("pose_frame"), **config["estimator"]}],
        remappings=[
            ("/camera_twist", get_arg("camera_twist_topic_name")),
            ("/pose", get_arg("pose_topic_name")),
            ("/pose_estimator/restart", get_arg("restart_topic_name")),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    pose_frame = LaunchConfiguration("pose_frame")

    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    pose_topic_name = LaunchConfiguration("pose_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("state_estimation_pkg")
    config_path = os.path.join(pkg_share, "config", "pose_estimator.toml")
    config = toml.load(config_path)

    # Initialize composable node
    pose_estimator_node = ComposableNode(
        package="state_estimation_pkg",
        plugin="PoseEstimator",
        parameters=[{"pose.frame": pose_frame, **config["estimator"]}],
        remappings=[
            ("/camera_twist", camera_twist_topic_name),
            ("/pose", pose_topic_name),
            ("/pose_estimator/restart", restart_topic_name),
        ],
    )

    # Initialize standalone composable node container
    pose_estimator_container = ComposableNodeContainer(
        package="rclcpp_components",
        name="pose_estimator_container",
        namespace="",
        executable="component_container",
        composable_node_descriptions=[pose_estimator_node],
        output="screen",
    )

    nodes_to_start = [pose_estimator_container]
    return LaunchDescription(declared_arguments + nodes_to_start)
