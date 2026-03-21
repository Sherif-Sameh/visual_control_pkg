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

    # ChArUco detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualize",
            default_value="false",
            description="Enable detection visualization topic. Default value is false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "dict_name",
            default_value="DICT_5X5_50",
            description="Name of the predefined marker dictionary. Default value is DICT_5X5_50.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_xs",
            default_value="10",
            description="Number of chessboard squares in X direction. Default value is 10.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_ys",
            default_value="10",
            description="Number of chessboard squares in Y direction. Default value is 10.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_sq_len",
            default_value="0.04",
            description="Chessboard square side length in meters. Default value is 0.04.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_mk_len",
            default_value="0.03",
            description="Marker side length in meters. Default value is 0.03.",
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
            "image_topic_name",
            default_value="/image",
            description="Input image (sensor_msgs/Image) topic name to use for visualizer.",
        )
    )
    return declared_arguments


def get_composable_node(**kwargs) -> ComposableNode:
    # Initialize argument defaults for missing arguments
    defaults = {a.name: a.default_value for a in declare_arguments()}

    # Load configuration from toml
    pkg_share = get_package_share_directory("vision_pkg")
    config_path = os.path.join(pkg_share, "config", "charuco_detector.toml")
    config = toml.load(config_path)

    # Initialize composable node
    get_arg = lambda name: kwargs.get(name, defaults[name])  # noqa: E731
    return ComposableNode(
        package="vision_pkg",
        plugin="CharucoDetector",
        parameters=[
            {
                "visualize": get_arg("visualize"),
                "dict.name": get_arg("dict_name"),
                "board.xs": get_arg("board_xs"),
                "board.ys": get_arg("board_ys"),
                "board.sq_len": get_arg("board_sq_len"),
                "board.mk_len": get_arg("board_mk_len"),
                **config["detector"],
            }
        ],
        remappings=[
            ("/image", get_arg("image_topic_name")),
            ("/camera_info", get_arg("camera_info_topic_name")),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    visualize = LaunchConfiguration("visualize")
    dict_name = LaunchConfiguration("dict_name")
    board_xs = LaunchConfiguration("board_xs")
    board_ys = LaunchConfiguration("board_ys")
    board_sq_len = LaunchConfiguration("board_sq_len")
    board_mk_len = LaunchConfiguration("board_mk_len")

    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    image_topic_name = LaunchConfiguration("image_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("vision_pkg")
    config_path = os.path.join(pkg_share, "config", "charuco_detector.toml")
    config = toml.load(config_path)

    # Initialize composable node
    charuco_detector_node = ComposableNode(
        package="vision_pkg",
        plugin="CharucoDetector",
        parameters=[
            {
                "visualize": visualize,
                "dict.name": dict_name,
                "board.xs": board_xs,
                "board.ys": board_ys,
                "board.sq_len": board_sq_len,
                "board.mk_len": board_mk_len,
                **config["detector"],
            }
        ],
        remappings=[("/camera_info", camera_info_topic_name), ("/image", image_topic_name)],
    )

    # Initialize standalone composable node container
    charuco_detector_container = ComposableNodeContainer(
        package="rclcpp_components",
        name="charuco_detector_container",
        namespace="",
        executable="component_container",
        composable_node_descriptions=[charuco_detector_node],
        output="screen",
    )

    nodes_to_start = [charuco_detector_container]
    return LaunchDescription(declared_arguments + nodes_to_start)
