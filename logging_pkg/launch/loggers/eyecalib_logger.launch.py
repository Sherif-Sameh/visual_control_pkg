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
            default_value="EyeCalib|ideal",
            description="Group name for run to use for WandB logger. Default is EyeCalib|ideal.",
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
            "joint_trajectory_topic_name",
            default_value="/joint_trajectory_controller/joint_trajectory",
            description="Joint trajectory (trajectory_msgs/JointTrajectory) topic name."
            " Default is /joint_trajectory_controller/joint_trajectory.",
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
            "setpoint_error_topic_name",
            default_value="/setpoint_error",
            description="Setpoint error (geometry_msgs/PoseArray) topic name."
            " Default is /setpoint_error.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/ros_logger/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /ros_logger/restart.",
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
    csv_dir = PathJoinSubstitution(
        [FindPackageShare("logging_pkg"), "../../../../logs/csv/eyecalib"]
    )
    wandb_dir = PathJoinSubstitution(
        [FindPackageShare("logging_pkg"), "../../../../logs/wandb/eyecalib"]
    )
    wandb_group = LaunchConfiguration("wandb_group")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    pose_error_topic_name = LaunchConfiguration("pose_error_topic_name")
    setpoint_error_topic_name = LaunchConfiguration("setpoint_error_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("logging_pkg")
    config_path = os.path.join(pkg_share, "config", "eyecalib_logger.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    eyecalib_logger_node = Node(
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
            ("/joint_states", joint_states_topic_name),
            ("/joint_trajectory", joint_trajectory_topic_name),
            ("/pose_error", pose_error_topic_name),
            ("/setpoint_error", setpoint_error_topic_name),
            ("/ros_logger/restart", restart_topic_name),
        ],
    )

    nodes_to_start = [eyecalib_logger_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
