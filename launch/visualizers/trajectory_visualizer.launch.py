from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Trajectory visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "target",
            default_value="['base_link']",
            description="Names of target frames for each tracked frame. Default is ['base_link'].",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "source",
            default_value="['ee_link']",
            description="Names of source frames for each tracked frame. Default is ['ee_link'].",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "length",
            default_value="30",
            description="Length of trajectory for each tracked frame. Default is 30.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "spacing",
            default_value="0.025",
            description="Spacing threshold for adding new point to tracked trajectories in meters."
            " Default is 0.025.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "width",
            default_value="0.025",
            description="Width of line strip for each tracked frame in meters. Default is 0.025.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "alpha",
            default_value="0.5",
            description="Opacity of line strip for each tracked frame. Default is 0.5.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "color",
            default_value="[0.0, 1.0, 0.0]",
            description="Concatenated list of RGB colors for each tracked frame."
            " Default is [0.0, 1.0, 0.0].",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "reset_topic_name",
            default_value="/trajectory_visualizer/reset",
            description="Reset (std_msgs/Empty) topic name. Default is /trajectory_visualizer/reset",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    target = LaunchConfiguration("target")
    source = LaunchConfiguration("source")
    length = LaunchConfiguration("length")
    spacing = LaunchConfiguration("spacing")
    width = LaunchConfiguration("width")
    alpha = LaunchConfiguration("alpha")
    color = LaunchConfiguration("color")

    reset_topic_name = LaunchConfiguration("reset_topic_name")

    # Initialize nodes to start
    trajectory_visualizer_node = Node(
        package="visual_control_pkg",
        executable="trajectory_visualizer.py",
        output="screen",
        parameters=[
            {
                "frame.target": target,
                "frame.source": source,
                "traj.length": length,
                "traj.spacing": spacing,
                "traj.width": width,
                "traj.alpha": alpha,
                "traj.color": color,
            }
        ],
        remappings=[("/trajectory_visualizer/reset", reset_topic_name)],
    )

    nodes_to_start = [trajectory_visualizer_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
