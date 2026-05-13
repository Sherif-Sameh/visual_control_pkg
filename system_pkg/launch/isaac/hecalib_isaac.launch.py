import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

# Contants
UR_TYPE = "ur10e"
EXECUTION_MODE = "calibration"
USE_ISAAC_CELL = "true"

BASE_FRAME = "base_link"
EE_FRAME = "tool0"
CAM_FRAME = "camera_color_optical_frame"
TCP_FRAME = CAM_FRAME
REF_FRAME = "charuco:0f"

PLANNER_MODE = "default"
POSE_MK_TGT = "[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]"
POSE_GT_HANDEYE = "[0.0, -0.04, 0.15, 0.9884, -0.1521, 0.0, 0.0]"

DICT_NAME = "DICT_5X5_50"
BOARD_XS = "10"
BOARD_YS = "10"
BOARD_SQ_LEN = "0.03"
BOARD_MK_LEN = "0.024"
BOARD_SIZE = "0.3"
TAG_ID = "0"

CAMERA_INFO_TOPIC_NAME = "/isaaclab/camera/camera_info"
POSE_REFERENCE_TOPIC_NAME = "/handeye_calibration/command"
PLANNED_TRAJECTORY_TOPIC_NAME = "/oc_planner/trajectory"
DETECTIONS_FILTERED_TOPIC_NAME = "/charuco_estimator/detections_filtered"
DETECTIONS_TOPIC_NAME = "/charuco_detector/detections"
IMAGE_TOPIC_NAME = "/isaaclab/camera/image_raw"
JOINT_STATES_TOPIC_NAME = "/isaaclab/joint_states"
JOINT_TRAJECTORY_TOPIC_NAME = "/isaaclab/joint_trajectory_controller/joint_trajectory"
POSE_ERROR_TOPIC_NAME = "/handeye_calibration/pose_error"
RESTART_TOPIC_NAME = "/isaaclab/reset"


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Control package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "controller",
            default_value="pbvs",
            description="Controller to use for tracking. Default is pbvs",
            choices=["pbvs"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="Enable verbose output from controller. Default value is false.",
        )
    )

    # State estimation package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "estimator",
            default_value="charuco",
            description="Name of estimator node to launch. Default value is charuco.",
            choices=["charuco", ""],
        )
    )

    # Vision package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "detector",
            default_value="charuco",
            description="Name of detector node to launch. Default value is charuco.",
            choices=["charuco", ""],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualize",
            default_value="false",
            description="Enable ChArUco detection visualization topic. Default value is false.",
        )
    )

    # Logging package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "logger",
            default_value="",
            description="Logger to launch. Default value '' is (empty string).",
            choices=["hecalib", ""],
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
            default_value="HECalib|default",
            description="Group name for run to use for WandB logger."
            " Default value is HECalib|default.",
        )
    )

    # Sweep package unique arguments (not shared with logging package)
    declared_arguments.append(
        DeclareLaunchArgument(
            "sweep",
            default_value="",
            description="Sweep to launch. Default value is '' (empty string).",
            choices=["hecalib", ""],
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
            default_value="isaac/hecalib.rviz",
            description="File name for the .rviz configuration file to load."
            " Default is isaac/hecalib.rviz.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualizers",
            default_value="r,t,p",
            description="Comma separated string of visualizers to enable. Use empty string to"
            " disable all. Default is 'r,t,p'.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "compose_charuco",
            default_value="true",
            description="Compose both ChArUco detector and estimator into the same node container."
            " Default is true.",
        )
    )
    return declared_arguments


def launch_setup(context) -> list[ComposableNodeContainer | IncludeLaunchDescription]:
    launch = []
    launch.append(_launch_control_pkg())
    launch.append(_launch_calibration_pkg())
    launch.append(_launch_state_estimation_pkg(context))
    launch.append(_launch_vision_pkg(context))
    launch.append(_launch_logging_pkg(context))
    launch.append(_launch_sweep_pkg())
    launch.append(_launch_visualization_pkg(context))
    if (charuco_container := _launch_charuco_container(context)) is not None:
        launch.append(charuco_container)
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


def _launch_control_pkg() -> IncludeLaunchDescription:
    controller = LaunchConfiguration("controller")
    verbose = LaunchConfiguration("verbose")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("control_pkg"), "launch", "control_pkg.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": UR_TYPE,
            "verbose": verbose,
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "cam_frame": CAM_FRAME,
            "tag_id": TAG_ID,
            "planner_mode": PLANNER_MODE,
            "tcp_frame": TCP_FRAME,
            "pose_mk_tgt": POSE_MK_TGT,
            "controller": controller,
            "pose_reference_topic_name": POSE_REFERENCE_TOPIC_NAME,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_FILTERED_TOPIC_NAME,
        }.items(),
    )


def _launch_calibration_pkg() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("calibration_pkg"), "launch", "calibration_pkg.launch.py"]
            )
        ),
        launch_arguments={
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "cam_frame": CAM_FRAME,
            "pose_gt_handeye": POSE_GT_HANDEYE,
            "calibration": "handeye_calibration",
            "detections_topic_name": DETECTIONS_FILTERED_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_state_estimation_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    estimator = LaunchConfiguration("estimator").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    compose_charuco = LaunchConfiguration("compose_charuco").perform(context)
    if controller not in ["pbvs"] or compose_charuco == "true":
        estimator = estimator.replace("charuco", "")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("state_estimation_pkg"),
                    "launch",
                    "state_estimation_pkg.launch.py",
                ]
            )
        ),
        launch_arguments={
            "marker_size": BOARD_SIZE,
            "estimator": estimator,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "camera_twist_topic_name": f"{controller}_controller/camera_twist",
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_vision_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    detector = LaunchConfiguration("detector").perform(context)
    visualize = LaunchConfiguration("visualize")
    controller = LaunchConfiguration("controller").perform(context)
    compose_charuco = LaunchConfiguration("compose_charuco").perform(context)
    if controller not in ["pbvs", "ibvs"] or compose_charuco == "true":
        detector = detector.replace("charuco", "")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("vision_pkg"), "launch", "vision_pkg.launch.py"])
        ),
        launch_arguments={
            "visualize": visualize,
            "dict_name": DICT_NAME,
            "board_xs": BOARD_XS,
            "board_ys": BOARD_YS,
            "board_sq_len": BOARD_SQ_LEN,
            "board_mk_len": BOARD_MK_LEN,
            "detector": detector,
            "image_topic_name": IMAGE_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
        }.items(),
    )


def _launch_logging_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    logger = LaunchConfiguration("logger")
    n_runs = LaunchConfiguration("n_runs")
    smooth = LaunchConfiguration("smooth")
    console = LaunchConfiguration("console")
    csv = LaunchConfiguration("csv")
    wandb = LaunchConfiguration("wandb")
    wandb_group = LaunchConfiguration("wandb_group")
    controller = LaunchConfiguration("controller").perform(context)

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
            "pose_error_topic_name": POSE_ERROR_TOPIC_NAME,
            "setpoint_error_topic_name": f"{controller}_controller/pose_error",
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_sweep_pkg() -> IncludeLaunchDescription:
    sweep = LaunchConfiguration("sweep")
    n_runs = LaunchConfiguration("n_runs")
    sweep_id = LaunchConfiguration("sweep_id")
    wandb_group = LaunchConfiguration("wandb_group")

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
            "pose_error_topic_name": POSE_ERROR_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_visualization_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    visualizers = LaunchConfiguration("visualizers").perform(context)
    visualizers = visualizers.replace("d", "")

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
            "source_frames": f"[{CAM_FRAME}]",
            "ref_frame": REF_FRAME,
            "visualizers": visualizers,
            "planned_trajectory_topic_name": PLANNED_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": "/joint_states",
            "image_topic_name": IMAGE_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_charuco_container(context: LaunchContext) -> ComposableNodeContainer | None:
    visualize = LaunchConfiguration("visualize").perform(context)
    compose_charuco = LaunchConfiguration("compose_charuco").perform(context)
    if compose_charuco == "false":
        return None
    controller = LaunchConfiguration("controller").perform(context)
    combined_kwargs = {
        "board_size": float(BOARD_SIZE),
        "visualize": visualize == "true",
        "dict_name": DICT_NAME,
        "board_xs": int(BOARD_XS),
        "board_ys": int(BOARD_YS),
        "board_sq_len": float(BOARD_SQ_LEN),
        "board_mk_len": float(BOARD_MK_LEN),
        "image_topic_name": IMAGE_TOPIC_NAME,
        "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
        "camera_twist_topic_name": f"{controller}_controller/camera_twist",
        "detections_topic_name": DETECTIONS_TOPIC_NAME,
    }
    return ComposableNodeContainer(
        package="rclcpp_components",
        name="charuco_container",
        namespace="",
        executable="component_container_mt",
        composable_node_descriptions=[
            _get_composable_node_description(
                "vision_pkg", "detectors/charuco_detector.launch.py", **combined_kwargs
            ),
            _get_composable_node_description(
                "state_estimation_pkg", "estimators/charuco_estimator.launch.py", **combined_kwargs
            ),
        ],
        output="screen",
    )


def _get_composable_node_description(pkg: str, path: str, **kwargs) -> ComposableNode:
    pkg_share = get_package_share_directory(pkg)
    node_launch_path = os.path.join(pkg_share, "launch", path)
    spec = importlib.util.spec_from_file_location(
        path.split("/")[-1].removesuffix(".launch.py"), node_launch_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_composable_node(**kwargs)
