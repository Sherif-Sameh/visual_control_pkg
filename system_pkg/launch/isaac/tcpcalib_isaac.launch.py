from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare

# Contants
UR_TYPE = "ur10e"
EXECUTION_MODE = "tracking"
USE_ISAAC_CELL = "true"

BASE_FRAME = "base_link"
EE_FRAME = "tool0"
CAM_FRAME = "camera_color_optical_frame"

POSE_GT_TCP = "[0.0, 0.006, 0.113, 0.988, 0.152, 0.0, 0.0]"
IMG_CENTER = "[300, 320]"

CAMERA_INFO_TOPIC_NAME = "/isaaclab/camera/camera_info"
IMAGE_TOPIC_NAME = "/isaaclab/camera/image_raw"
DEPTH_TOPIC_NAME = "/isaaclab/camera/depth_raw"
JOINT_STATES_TOPIC_NAME = "/isaaclab/joint_states"
JOINT_TRAJECTORY_TOPIC_NAME = "/isaaclab/joint_trajectory_controller/joint_trajectory"
RESTART_TOPIC_NAME = "/isaaclab/reset"


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Calibration package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "calibration",
            default_value="tcp_calibration_p3d",
            description="Name of calibration node to launch. Default is tcp_calibration_p3d.",
            choices=["tcp_calibration_p3d"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "modalities",
            default_value="['depth']",
            description="Rendering modalities to use for TCP pose optimization. Options include"
            " 'silhouette' and 'depth' only. Default is ['depth'].",
        )
    )

    # Logging package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "logger",
            default_value="",
            description="Logger to launch. Default value '' is (empty string).",
            choices=["tcpcalib_p3d", ""],
        )
    )
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
            default_value="",
            description="Group name for run to use for WandB logger. If an empty string, the group"
            " name is set to uppercase(logger/sweep).replace(_,|)|default. Default value empty string.",
        )
    )

    # Sweep package unique arguments (not shared with logging package)
    declared_arguments.append(
        DeclareLaunchArgument(
            "sweep",
            default_value="",
            description="Sweep to launch. Default value '' is (empty string).",
            choices=["tcpcalib_p3d", ""],
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

    # Visualization package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz", default_value="true", description="Launch RViz. Default is true."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="isaac/isaac.rviz",
            description="File name for the .rviz configuration file to load."
            " Default is isaac/isaac.rviz.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualizers",
            default_value="r",
            description="Comma separated string of visualizers to enable. Use empty string to"
            " disable all. Default is 'r'.",
        )
    )

    return declared_arguments


def launch_setup(context) -> list[ComposableNodeContainer | IncludeLaunchDescription]:
    launch = []
    launch.append(_launch_calibration_pkg())
    launch.append(_launch_logging_pkg(context))
    launch.append(_launch_sweep_pkg(context))
    launch.append(_launch_visualization_pkg(context))
    return launch


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Add opaque functions
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + opaque_functions)


##
# Private functions
##


def _launch_calibration_pkg() -> IncludeLaunchDescription:
    calibration = LaunchConfiguration("calibration")
    modalities = LaunchConfiguration("modalities")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("calibration_pkg"), "launch", "calibration_pkg.launch.py"]
            )
        ),
        launch_arguments={
            "pose_gt_tcp": POSE_GT_TCP,
            "img_center": IMG_CENTER,
            "modalities": modalities,
            "calibration": calibration,
            "image_topic_name": IMAGE_TOPIC_NAME,
            "depth_topic_name": DEPTH_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_logging_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    logger = LaunchConfiguration("logger")
    n_runs = LaunchConfiguration("n_runs")
    smooth = LaunchConfiguration("smooth")
    console = LaunchConfiguration("console")
    csv = LaunchConfiguration("csv")
    wandb = LaunchConfiguration("wandb")
    wandb_group = LaunchConfiguration("wandb_group").perform(context)
    calibration = LaunchConfiguration("calibration").perform(context)
    if wandb_group == "":
        wandb_group = f"TCPCalib|{calibration.split('_')[-1].upper()}|default"

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("logging_pkg"), "launch", "logging_pkg.launch.py"]
            )
        ),
        launch_arguments={
            "n_runs": n_runs,
            "smooth": smooth,
            "console": console,
            "csv": csv,
            "wandb": wandb,
            "wandb_group": wandb_group,
            "logger": logger,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "pose_error_topic_name": f"tcp_calibration_{calibration.split('_')[-1]}/pose_error",
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_sweep_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    sweep = LaunchConfiguration("sweep")
    n_runs = LaunchConfiguration("n_runs")
    sweep_id = LaunchConfiguration("sweep_id")
    wandb_group = LaunchConfiguration("wandb_group").perform(context)
    calibration = LaunchConfiguration("calibration").perform(context)
    if wandb_group == "":
        wandb_group = f"TCPCalib|{calibration.split('_')[-1].upper()}|default"

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("sweep_pkg"), "launch", "sweep_pkg.launch.py"])
        ),
        launch_arguments={
            "n_runs": n_runs,
            "sweep_id": sweep_id,
            "wandb_group": wandb_group,
            "sweep": sweep,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "pose_error_topic_name": f"tcp_calibration_{calibration.split('_')[-1]}/pose_error",
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_visualization_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    visualizers = LaunchConfiguration("visualizers").perform(context)

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("visualization_pkg"), "launch", "visualization_pkg.launch.py"]
            )
        ),
        launch_arguments={
            "rviz": rviz,
            "rviz_config": rviz_config,
            "ur_type": UR_TYPE,
            "execution_mode": EXECUTION_MODE,
            "use_isaac_cell": USE_ISAAC_CELL,
            "target_frames": f"[{BASE_FRAME}]",
            "source_frames": f"[{EE_FRAME}]",
            "visualizers": visualizers,
            "joint_states_topic_name": "/joint_states",
            "image_topic_name": IMAGE_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )
