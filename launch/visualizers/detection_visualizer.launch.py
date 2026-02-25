from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Trajectory visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "center_radius",
            default_value="5",
            description="Radius of tag center circle in pixels. Default is 5.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "border_thickness",
            default_value="2",
            description="Thickness of tag surrounding border in pixels. Default is 2.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "font_thickness",
            default_value="2",
            description="Thickness of tag ID font in pixels. Default is 2.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "font_scale",
            default_value="1.0",
            description="Scale factor for tag ID font. Default is 1.0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "offset_x",
            default_value="-35",
            description="Offset to apply to tag ID font location relative to its center for x(u)-axis."
            " Default is -35.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "offset_y",
            default_value="-35",
            description="Offset to apply to tag ID font location relative to its center for y(v)-axis."
            " Default is -35.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "image_topic_name",
            default_value="/image",
            description="Input image (sensor_msgs/Image) topic name to use for visualizer.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="Detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray) topic"
            " name to use for visualizer.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    center_radius = LaunchConfiguration("center_radius")
    border_thickness = LaunchConfiguration("border_thickness")
    font_thickness = LaunchConfiguration("font_thickness")
    font_scale = LaunchConfiguration("font_scale")
    offset_x = LaunchConfiguration("offset_x")
    offset_y = LaunchConfiguration("offset_y")

    image_topic_name = LaunchConfiguration("image_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Initialize nodes to start
    detection_visualizer_node = Node(
        package="visual_control_pkg",
        executable="detection_visualizer.py",
        output="screen",
        parameters=[
            {
                "center.radius": center_radius,
                "border.thickness": border_thickness,
                "font.thickness": font_thickness,
                "font.scale": font_scale,
                "font.offset_x": offset_x,
                "font.offset_y": offset_y,
            }
        ],
        remappings=[("/image", image_topic_name), ("/detections", detections_topic_name)],
    )

    nodes_to_start = [detection_visualizer_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
