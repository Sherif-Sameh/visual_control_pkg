from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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
            default_value="group",
            description="Group name for run to use for WandB logger. Default value group.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "logger",
            default_value="''",
            description="Logger to launch. Default value '' is (empty string).",
            choices=["pbvs", "ibvs", "''"],
        )
    )
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


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    logger = LaunchConfiguration("logger").perform(context)
    # Launch chosen logger
    match logger:
        case "pbvs":
            return [_include_pbvs_logger()]
        case "ibvs":
            return [_include_ibvs_logger()]
        case _:
            return []


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Add opaque functions
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + opaque_functions)


##
# Private functions
##


def _include_pbvs_logger() -> IncludeLaunchDescription:
    n_runs = LaunchConfiguration("n_runs")
    smooth = LaunchConfiguration("smooth")
    console = LaunchConfiguration("console")
    csv = LaunchConfiguration("csv")
    wandb = LaunchConfiguration("wandb")
    wandb_group = LaunchConfiguration("wandb_group")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    pose_error_topic_name = LaunchConfiguration("pose_error_topic_name")
    setpoint_error_topic_name = LaunchConfiguration("setpoint_error_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("logging_pkg"), "launch", "loggers", "pbvs_logger.launch.py"]
            )
        ),
        launch_arguments={
            "n_runs": n_runs,
            "smooth": smooth,
            "console": console,
            "csv": csv,
            "wandb": wandb,
            "wandb_group": wandb_group,
            "joint_states_topic_name": joint_states_topic_name,
            "joint_trajectory_topic_name": joint_trajectory_topic_name,
            "pose_error_topic_name": pose_error_topic_name,
            "setpoint_error_topic_name": setpoint_error_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_ibvs_logger() -> IncludeLaunchDescription:
    n_runs = LaunchConfiguration("n_runs")
    smooth = LaunchConfiguration("smooth")
    console = LaunchConfiguration("console")
    csv = LaunchConfiguration("csv")
    wandb = LaunchConfiguration("wandb")
    wandb_group = LaunchConfiguration("wandb_group")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    pose_error_topic_name = LaunchConfiguration("pose_error_topic_name")
    setpoint_error_topic_name = LaunchConfiguration("setpoint_error_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("logging_pkg"), "launch", "loggers", "ibvs_logger.launch.py"]
            )
        ),
        launch_arguments={
            "n_runs": n_runs,
            "smooth": smooth,
            "console": console,
            "csv": csv,
            "wandb": wandb,
            "wandb_group": wandb_group,
            "joint_states_topic_name": joint_states_topic_name,
            "joint_trajectory_topic_name": joint_trajectory_topic_name,
            "pose_error_topic_name": pose_error_topic_name,
            "setpoint_error_topic_name": setpoint_error_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )
