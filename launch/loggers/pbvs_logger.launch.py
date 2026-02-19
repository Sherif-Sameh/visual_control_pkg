from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # ROS Logger arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "period",
            default_value="0.1",
            description="Period for logging callback in seconds. Default is 0.1.",
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
            "console_n_log",
            default_value="5",
            description="Logging interval for console logger relative to logging period."
            " Default is 5.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "console_n_flush",
            default_value="5",
            description="Flushing interval for console logger relative to logging period."
            " Default is 5.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "csv_n_log",
            default_value="1",
            description="Logging interval for CSV logger relative to logging period."
            " Default is 1.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "csv_n_flush",
            default_value="10",
            description="Flushing interval for CSV logger relative to logging period."
            " Default is 10.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "wandb_n_log",
            default_value="2",
            description="Logging interval for WandB logger relative to logging period."
            " Default is 2.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "wandb_n_flush",
            default_value="12",
            description="Flushing interval for WandB logger relative to logging period."
            " Default is 12.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    period = LaunchConfiguration("period")
    console = LaunchConfiguration("console")
    csv = LaunchConfiguration("csv")
    wandb = LaunchConfiguration("wandb")
    console_n_log = LaunchConfiguration("console_n_log")
    console_n_flush = LaunchConfiguration("console_n_flush")
    csv_n_log = LaunchConfiguration("csv_n_log")
    csv_n_flush = LaunchConfiguration("csv_n_flush")
    csv_dir = PathJoinSubstitution(
        [
            FindPackageShare("visual_control_pkg"),
            "../../../../logs/csv",
        ]
    )
    wandb_n_log = LaunchConfiguration("wandb_n_log")
    wandb_n_flush = LaunchConfiguration("wandb_n_flush")
    wandb_dir = PathJoinSubstitution(
        [
            FindPackageShare("visual_control_pkg"),
            "../../../../logs/wandb",
        ]
    )
    wandb_params = [
        "apriltag/size",
        "apriltag/tile_size",
        "apriltag/backends",
        "pbvs_controller/tag.tag_ids",
        "pbvs_controller/ik.eps",
        "pbvs_controller/ik.lambda",
        "pbvs_controller/ik.max_iters",
        "pbvs_controller/ik.weight_js",
        "pbvs_controller/robot.max_tvel",
        "pbvs_controller/robot.max_rvel",
        "pbvs_controller/robot.max_vel_sf",
        "pbvs_controller/robot.max_qdot",
        "pbvs_controller/ctrl.conv_ttol",
        "pbvs_controller/ctrl.conv_rtol",
        "pbvs_controller/ctrl.lambda",
    ]

    # Initialize nodes to start
    pbvs_logger_node = Node(
        package="visual_control_pkg",
        executable="ros_logger.py",
        output="screen",
        parameters=[
            {
                "timer_period": period,
                "param_servers": [
                    "apriltag/get_parameters",
                    "pbvs_controller/get_parameters",
                ],
                "log.console": console,
                "log.csv": csv,
                "log.wandb": wandb,
                "console.n_log": console_n_log,
                "console.n_flush": console_n_flush,
                "console.filter": ["PE", "SE"],
                "console.config.precision": 3,
                "console.config.separator": "  ",
                "console.config.sign": "+",
                "csv.n_log": csv_n_log,
                "csv.n_flush": csv_n_flush,
                "csv.filter": [""],
                "csv.dir": csv_dir,
                "wandb.n_log": wandb_n_log,
                "wandb.n_flush": wandb_n_flush,
                "wandb.filter": [""],
                "wandb.config.entity": "u1999168-girona",
                "wandb.config.project": "visual_control|PBVS",
                "wandb.config.group": "test",
                "wandb.config.dir": wandb_dir,
                "wandb.config.params": wandb_params,
            }
        ],
        remappings=[
            ("/joint_states", "/isaaclab/joint_states"),
            (
                "/joint_trajectory",
                "/isaaclab/joint_trajectory_controller/joint_trajectory",
            ),
            ("/pose_error", "/isaaclab/pose_error"),
            ("/setpoint_error", "/pbvs_controller/pose_error"),
            ("/ros_logger/restart", "/isaaclab/reset"),
        ],
    )

    nodes_to_start = [pbvs_logger_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
