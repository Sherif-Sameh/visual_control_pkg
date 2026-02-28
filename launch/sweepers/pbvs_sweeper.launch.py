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

    # ROS Sweeper arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "n_runs",
            default_value="0",
            description="Number of runs for the hyperparameter sweep. The node will run infinitely"
            " for any value <= 0. Default is 0 (i.e. infinite).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "sweep_id",
            default_value="''",
            description="Sweep ID for the WandB sweep. Should only be specified of continuing an"
            " existing sweep. Otherwise, it should be left empty. Default is '' (empty string).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "wandb_group",
            default_value="PBVS",
            description="Group name for run to use for WandB logger. Default is PBVS.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    n_runs = LaunchConfiguration("n_runs")
    sweep_id = LaunchConfiguration("sweep_id")
    wandb_group = LaunchConfiguration("wandb_group")
    wandb_dir = PathJoinSubstitution(
        [FindPackageShare("visual_control_pkg"), "../../../../logs/wandb"]
    )

    # Load configuration from toml
    pkg_share = get_package_share_directory("visual_control_pkg")
    config_path = os.path.join(pkg_share, "config", "sweepers", "pbvs_sweeper.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    pbvs_sweeper_node = Node(
        package="visual_control_pkg",
        executable="ros_sweeper.py",
        output="screen",
        parameters=[
            {
                "n_runs": n_runs,
                "sweep.id": sweep_id,
                "sweep.config": config_path,
                "wandb.config.group": wandb_group,
                "wandb.config.dir": wandb_dir,
                **config["launch"],
            }
        ],
        remappings=[
            ("/joint_states", "/isaaclab/joint_states"),
            ("/pose_error", "/isaaclab/pose_error"),
            ("/ros_sweeper/restart", "/isaaclab/reset"),
        ],
    )

    nodes_to_start = [pbvs_sweeper_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
