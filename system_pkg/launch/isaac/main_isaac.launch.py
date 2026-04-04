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
EXECUTION_MODE = "tracking"
USE_ISAAC_CELL = "true"

BASE_FRAME = "base_link"
EE_FRAME = "tool0"
CAM_FRAME = "camera_color_optical_frame"

POSE_GT_TCP = "[0.0, 0.006, 0.113, 0.988, 0.152, 0.0, 0.0]"
IMG_CENTER = "[300, 320]"

TAG_FAMILY = "tag36h11"
TAG_IDS = "[0, 1]"
TAG_SIZE = "0.08"

CAMERA_INFO_TOPIC_NAME = "/isaaclab/camera/camera_info"
DESIRED_TRAJECTORY_TOPIC_NAME = "/isaaclab/command"
DETECTIONS_FILTERED_TOPIC_NAME = "/apriltag_estimator/detections_filtered"
DETECTIONS_TOPIC_NAME = "/apriltag_detector/detections"
IMAGE_TOPIC_NAME = "/isaaclab/camera/image_raw"
DEPTH_TOPIC_NAME = "/isaaclab/camera/depth_raw"
JOINT_STATES_TOPIC_NAME = "/isaaclab/joint_states"
JOINT_TRAJECTORY_TOPIC_NAME = "/isaaclab/joint_trajectory_controller/joint_trajectory"
POSE_ERROR_TOPIC_NAME = "/isaaclab/pose_error"
RESTART_TOPIC_NAME = "/isaaclab/reset"


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Control package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "controller",
            default_value="pbvs",
            description="Controller to use for tracking. Default is pbvs",
            choices=["pbvs", "ibvs", "pose"],
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
            default_value="apriltag",
            description="Name of estimator node to launch. Default value is apriltag.",
            choices=["apriltag", ""],
        )
    )

    # Vision package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "detector",
            default_value="apriltag",
            description="Name of detector node to launch. Default value is apriltag.",
            choices=["apriltag", ""],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "backends",
            default_value="CPU",
            description="Backend to perform AprilTag detection with. Default value is CPU.",
            choices=["CUDA", "CPU", "PVA"],
        )
    )

    # Logging package arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "logger",
            default_value="",
            description="Logger to launch. Default value '' is (empty string).",
            choices=["pbvs", "ibvs", ""],
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
            " name is set to uppercase(controller)|default. Default value empty string.",
        )
    )

    # Sweep package unique arguments (not shared with logging package)
    declared_arguments.append(
        DeclareLaunchArgument(
            "sweep",
            default_value="",
            description="Sweep to launch. Default value '' is (empty string).",
            choices=["pbvs", "ibvs", ""],
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
            default_value="r,t",
            description="Comma separated string of visualizers to enable. Use empty string to"
            " disable all. Default is 'r,t'.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualize_raw_tags",
            default_value="false",
            description="Visualize raw AprilTag detections instead of filtered ones."
            " Default is false.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "compose_apriltag",
            default_value="true",
            description="Compose both AprilTag detector and estimator into the same node container."
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
    launch.append(_launch_sweep_pkg(context))
    launch.append(_launch_visualization_pkg(context))
    if (apriltag_container := _launch_apriltag_container(context)) is not None:
        launch.append(apriltag_container)
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
            "tag_size": TAG_SIZE,
            "tag_ids": TAG_IDS,
            "controller": controller,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_FILTERED_TOPIC_NAME,
            "desired_trajectory_topic_name": DESIRED_TRAJECTORY_TOPIC_NAME,
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
            "pose_gt_tcp": POSE_GT_TCP,
            "img_center": IMG_CENTER,
            "calibration": "tcp_calibration_p3d",
            "image_topic_name": IMAGE_TOPIC_NAME,
            "depth_topic_name": DEPTH_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_state_estimation_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    estimator = LaunchConfiguration("estimator").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    compose_apriltag = LaunchConfiguration("compose_apriltag").perform(context)
    if controller not in ["pbvs", "ibvs"] or compose_apriltag == "true":
        estimator = estimator.replace("apriltag", "")

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
            "marker_size": TAG_SIZE,
            "estimator": estimator,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "camera_twist_topic_name": f"{controller}_controller/camera_twist",
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
        }.items(),
    )


def _launch_vision_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    detector = LaunchConfiguration("detector").perform(context)
    backends = LaunchConfiguration("backends")
    controller = LaunchConfiguration("controller").perform(context)
    compose_apriltag = LaunchConfiguration("compose_apriltag").perform(context)
    if controller not in ["pbvs", "ibvs"] or compose_apriltag == "true":
        detector = detector.replace("apriltag", "")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("vision_pkg"), "launch", "vision_pkg.launch.py"])
        ),
        launch_arguments={
            "tag_size": TAG_SIZE,
            "tag_family": TAG_FAMILY,
            "backends": backends,
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
    wandb_group = LaunchConfiguration("wandb_group").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    if wandb_group == "":
        wandb_group = f"{controller.upper()}|default"

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


def _launch_sweep_pkg(context: LaunchContext) -> IncludeLaunchDescription:
    sweep = LaunchConfiguration("sweep")
    n_runs = LaunchConfiguration("n_runs")
    sweep_id = LaunchConfiguration("sweep_id")
    wandb_group = LaunchConfiguration("wandb_group").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    if wandb_group == "":
        wandb_group = f"{controller.upper()}|default"

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
    visualize_raw_tags = LaunchConfiguration("visualize_raw_tags").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    if controller not in ["pbvs", "ibvs"]:
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
            "source_frames": f"[{EE_FRAME}]",
            "visualizers": visualizers,
            "joint_states_topic_name": "/joint_states",
            "image_topic_name": IMAGE_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_TOPIC_NAME
            if visualize_raw_tags == "true"
            else DETECTIONS_FILTERED_TOPIC_NAME,
            "restart_topic_name": RESTART_TOPIC_NAME,
        }.items(),
    )


def _launch_apriltag_container(context: LaunchContext) -> ComposableNodeContainer | None:
    backends = LaunchConfiguration("backends").perform(context)
    compose_apriltag = LaunchConfiguration("compose_apriltag").perform(context)
    if compose_apriltag == "false":
        return None
    controller = LaunchConfiguration("controller").perform(context)
    combined_kwargs = {
        "tag_size": float(TAG_SIZE),
        "tag_family": TAG_FAMILY,
        "backends": backends,
        "image_topic_name": IMAGE_TOPIC_NAME,
        "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
        "camera_twist_topic_name": f"{controller}_controller/camera_twist",
        "detections_topic_name": DETECTIONS_TOPIC_NAME,
    }
    return ComposableNodeContainer(
        package="rclcpp_components",
        name="apriltag_container",
        namespace="",
        executable="component_container_mt",
        composable_node_descriptions=[
            _get_composable_node_description(
                "vision_pkg", "detectors/apriltag_detector.launch.py", **combined_kwargs
            ),
            _get_composable_node_description(
                "state_estimation_pkg", "estimators/apriltag_estimator.launch.py", **combined_kwargs
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
