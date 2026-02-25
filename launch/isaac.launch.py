from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription

# Contants
UR_TYPE = "ur10e"
BASE_FRAME = "base_link"
EE_FRAME = "wrist_3_link"
CAM_FRAME = "camera_color_optical_frame"

TAG_FAMILY = "tag36h11"
TAG_IDS = "[0, 1]"
TAG_SIZE = "0.08"

CAMERA_INFO_TOPIC_NAME = "/isaaclab/camera/camera_info"
DETECTIONS_TOPIC_NAME = "/detector/tag_detections"
IMAGE_TOPIC_NAME = "/isaaclab/camera/image_raw"
JOINT_STATES_TOPIC_NAME = "/isaaclab/joint_states"
JOINT_TRAJECTORY_TOPIC_NAME = "/isaaclab/joint_trajectory_controller/joint_trajectory"
RESET_TOPIC_NAME = "/isaaclab/reset"


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Isaac ROS Apriltag detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "backends",
            default_value="CPU",
            description="Backend to perform detection with. Default value is CPU.",
            choices=["CUDA", "CPU", "PVA"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_debugger",
            default_value="false",
            description="Enable the Apriltag detector debugger node for publishing"
            " additional visualizations. Default value is false.",
        )
    )

    # Trajectory visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualizer",
            default_value="t",
            description="Comma separated string of visualization to enable. Use empty string with"
            " any character (e.g., ' ') to disable all. Default is t.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "length",
            default_value="30",
            description="Length of visualized trajectory for end-effector. Default is 30.",
        )
    )

    # Controller arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "controller",
            default_value="pbvs",
            description="Controller to use for tracking. Default is pbvs",
            choices=["pbvs", "ibvs"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="Enable verbose output from PBVS controller. Default value is false.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz", default_value="true", description="Launch RViz. Defaults to true."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="isaac.rviz",
            description="File name for the .rviz configuration file to load."
            " Defaults to isaac.rviz.",
        )
    )
    return declared_arguments


def launch_setup(context) -> list[IncludeLaunchDescription]:
    launch = []
    launch += _launch_setup_visualizer(context)
    launch += _launch_setup_controller(context)
    return launch


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    backends = LaunchConfiguration("backends")
    use_debugger = LaunchConfiguration("use_debugger")

    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    # Include launch files
    view_ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "view_robot",
                    "view_ur.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": UR_TYPE,
            "rviz": rviz,
            "rviz_config": rviz_config,
            "joint_states_topic_name": "/joint_states",
        }.items(),
    )

    isaac_ros_apriltag_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "detectors",
                    "isaac_ros_apriltag.launch.py",
                ]
            )
        ),
        launch_arguments={
            "size": TAG_SIZE,
            "tag_family": TAG_FAMILY,
            "backends": backends,
            "use_debugger": use_debugger,
            "image_topic_name": IMAGE_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
        }.items(),
    )

    launch_files = [view_ur_launch, isaac_ros_apriltag_launch]

    # Add opaque functions
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + launch_files + opaque_functions)


##
# Private functions
##


def _launch_setup_visualizer(context) -> list[IncludeLaunchDescription]:
    visualizer = LaunchConfiguration("visualizer").perform(context)
    visualizer = visualizer.replace(" ", "").split(",")
    launch = []
    # Launch chosen visualizers
    if "t" in visualizer:
        length = LaunchConfiguration("length")
        trajectory_visualizer_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("visual_control_pkg"),
                        "launch",
                        "visualizers",
                        "trajectory_visualizer.launch.py",
                    ]
                )
            ),
            launch_arguments={
                "target": f"[{BASE_FRAME}]",
                "source": f"[{EE_FRAME}]",
                "length": length,
                "reset_topic_name": RESET_TOPIC_NAME,
            }.items(),
        )
        launch.append(trajectory_visualizer_launch)
    return launch


def _launch_setup_controller(context) -> list[IncludeLaunchDescription]:
    controller = LaunchConfiguration("controller").perform(context)
    assert controller in ["pbvs", "ibvs"]
    # Launch chosen controller
    match controller:
        case "pbvs":
            return [_include_controller_pbvs()]
        case "ibvs":
            return [_include_controller_ibvs()]


def _include_controller_pbvs() -> IncludeLaunchDescription:
    verbose = LaunchConfiguration("verbose")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "controllers",
                    "pbvs_controller.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": UR_TYPE,
            "verbose": verbose,
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "cam_frame": CAM_FRAME,
            "tag_family": TAG_FAMILY,
            "tag_ids": TAG_IDS,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
        }.items(),
    )


def _include_controller_ibvs() -> IncludeLaunchDescription:
    verbose = LaunchConfiguration("verbose")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "controllers",
                    "ibvs_controller.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": UR_TYPE,
            "verbose": verbose,
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "cam_frame": CAM_FRAME,
            "tag_family": TAG_FAMILY,
            "tag_size": TAG_SIZE,
            "tag_ids": TAG_IDS,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
        }.items(),
    )
