import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # ROS Logger arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "n_runs",
            default_value="0",
            description="Number of runs (restarts) for logger to run. The logger will run"
            " infinitely for any value <= 0. Default is 0 (i.e. infinite).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "smooth",
            default_value="false",
            description="Enable metric smoothing through averaging between timer callbacks."
            " Default is false (i.e. log only the last value before callback).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "console",
            default_value="false",
            description="Enable logging output to the console. Default is false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "csv",
            default_value="false",
            description="Enable logging output to the a CSV file. Default is false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "wandb",
            default_value="false",
            description="Enable logging output to the WandB. Default is false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "wandb_group",
            default_value="PBVS|ideal",
            description="Group name for run to use for WandB logger. Default is PBVS|ideal.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    n_runs = LaunchConfiguration("n_runs")
    smooth = LaunchConfiguration("smooth")
    console = LaunchConfiguration("console")
    csv = LaunchConfiguration("csv")
    wandb = LaunchConfiguration("wandb")
    csv_dir = PathJoinSubstitution([FindPackageShare("logging_pkg"), "../../../../logs/csv/pbvs"])
    wandb_dir = PathJoinSubstitution(
        [FindPackageShare("logging_pkg"), "../../../../logs/wandb/pbvs"]
    )
    wandb_group = LaunchConfiguration("wandb_group")

    # Load configuration from toml
    pkg_share = get_package_share_directory("logging_pkg")
    config_path = os.path.join(pkg_share, "config", "pbvs_logger.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    pbvs_logger_node = Node(
        package="logging_pkg",
        executable="ros_logger.py",
        output="screen",
        parameters=[
            {
                "n_runs": n_runs,
                "smooth": smooth,
                "log.console": console,
                "log.csv": csv,
                "log.wandb": wandb,
                "csv.dir": csv_dir,
                "wandb.config.group": wandb_group,
                "wandb.config.dir": wandb_dir,
                **config["logger"],
            }
        ],
        remappings=[
            ("/joint_states", "/isaaclab/joint_states"),
            ("/joint_trajectory", "/isaaclab/joint_trajectory_controller/joint_trajectory"),
            ("/pose_error", "/isaaclab/pose_error"),
            ("/setpoint_error", "/pbvs_controller/pose_error"),
            ("/ros_logger/restart", "/isaaclab/reset"),
        ],
    )

    nodes_to_start = [pbvs_logger_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
