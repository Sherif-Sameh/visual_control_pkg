import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
            default_value="",
            description="Sweep ID for the WandB sweep. Should only be specified of continuing an"
            " existing sweep. Otherwise, it should be left empty. Default is '' (empty string).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "wandb_group",
            default_value="EyeDtc|ideal",
            description="Group name for run to use for WandB logger. Default is EyeDtc|ideal.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "joint_states_topic_name",
            default_value="/joint_states",
            description="Joint states (sensor_msgs/JointState) topic name."
            " Default is /joint_states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_error_topic_name",
            default_value="/pose_error",
            description="Pose error (geometry_msgs/PoseStamped) topic name."
            " Default is /pose_error.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/ros_sweeper/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /ros_sweeper/restart.",
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
    wandb_dir = PathJoinSubstitution([FindPackageShare("sweep_pkg"), "../../../../logs/wandb"])

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    pose_error_topic_name = LaunchConfiguration("pose_error_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("sweep_pkg")
    config_path = os.path.join(pkg_share, "config", "eyedtc_sweep.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    eyedtc_sweeper_node = Node(
        package="sweep_pkg",
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
            ("/joint_states", joint_states_topic_name),
            ("/pose_error", pose_error_topic_name),
            ("/ros_sweeper/restart", restart_topic_name),
        ],
    )

    nodes_to_start = [eyedtc_sweeper_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
