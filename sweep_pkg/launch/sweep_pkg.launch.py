from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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
            default_value="''",
            description="Sweep ID for the WandB sweep. Should only be specified of continuing an"
            " existing sweep. Otherwise, it should be left empty. Default is '' (empty string).",
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
            "sweep",
            default_value="''",
            description="Sweep to launch. Default value '' is (empty string).",
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


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    sweep = LaunchConfiguration("sweep").perform(context)
    # Launch chosen sweep
    match sweep:
        case "pbvs":
            return [_include_pbvs_sweep()]
        case "ibvs":
            return [_include_ibvs_sweep()]
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


def _include_pbvs_sweep() -> IncludeLaunchDescription:
    n_runs = LaunchConfiguration("n_runs")
    sweep_id = LaunchConfiguration("sweep_id")
    wandb_group = LaunchConfiguration("wandb_group")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    pose_error_topic_name = LaunchConfiguration("pose_error_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("sweep_pkg"), "launch", "sweeps", "pbvs_sweep.launch.py"]
            )
        ),
        launch_arguments={
            "n_runs": n_runs,
            "sweep_id": sweep_id,
            "wandb_group": wandb_group,
            "joint_states_topic_name": joint_states_topic_name,
            "pose_error_topic_name": pose_error_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_ibvs_sweep() -> IncludeLaunchDescription:
    n_runs = LaunchConfiguration("n_runs")
    sweep_id = LaunchConfiguration("sweep_id")
    wandb_group = LaunchConfiguration("wandb_group")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    pose_error_topic_name = LaunchConfiguration("pose_error_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("sweep_pkg"), "launch", "sweeps", "ibvs_sweep.launch.py"]
            )
        ),
        launch_arguments={
            "n_runs": n_runs,
            "sweep_id": sweep_id,
            "wandb_group": wandb_group,
            "joint_states_topic_name": joint_states_topic_name,
            "pose_error_topic_name": pose_error_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )
