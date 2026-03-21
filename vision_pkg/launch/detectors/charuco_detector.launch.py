import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

    # Initialize nodes to start
    charuco_detector_node = Node(
        package="vision_pkg",
        executable="charuco_detector",
        output="screen",
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

    nodes_to_start = [charuco_detector_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
