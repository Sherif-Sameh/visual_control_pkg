from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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

    # Visualization arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualizer",
            default_value="r,t",
            description="Comma separated string of visualizations to enable. Use empty string with"
            " any character (e.g., ' ') to disable all. Default is 'r,t'.",
        )
    )

    # Controller arguments
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


def launch_setup(context) -> list[Node | IncludeLaunchDescription]:
    launch = []
    launch += _launch_setup_rviz(context)
    launch += _launch_setup_visualizer(context)
    launch += _launch_setup_controller(context)
    return launch


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    backends = LaunchConfiguration("backends")

    # Include launch files
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
            "image_topic_name": IMAGE_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
        }.items(),
    )

    launch_files = [isaac_ros_apriltag_launch]

    # Add opaque functions
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + launch_files + opaque_functions)


##
# Private functions
##


def _launch_setup_rviz(context) -> list[Node]:
    rviz = LaunchConfiguration("rviz").perform(context)
    if rviz == "true":
        rviz_config = LaunchConfiguration("rviz_config")
        rviz_config_file = PathJoinSubstitution(
            [FindPackageShare("visual_control_pkg"), "rviz", rviz_config]
        )
        return [
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="log",
                arguments=["-d", rviz_config_file],
            )
        ]
    return []


def _launch_setup_visualizer(context) -> list[IncludeLaunchDescription]:
    visualizer = LaunchConfiguration("visualizer").perform(context)
    visualizer = visualizer.replace(" ", "").split(",")
    launch = []
    # Launch chosen visualizers
    if "r" in visualizer:
        launch.append(_include_visualizer_robot())
    if "t" in visualizer:
        launch.append(_include_visualizer_trajectory())
    if "d" in visualizer:
        launch.append(_include_visualizer_detection())
    return launch


def _include_visualizer_robot() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "visualizers",
                    "robot_visualizer.launch.py",
                ]
            )
        ),
        launch_arguments={"ur_type": UR_TYPE, "joint_states_topic_name": "/joint_states"}.items(),
    )


def _include_visualizer_trajectory() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
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
            "reset_topic_name": RESET_TOPIC_NAME,
        }.items(),
    )


def _include_visualizer_detection() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "visualizers",
                    "detection_visualizer.launch.py",
                ]
            )
        ),
        launch_arguments={
            "image_topic_name": IMAGE_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
        }.items(),
    )


def _launch_setup_controller(context) -> list[IncludeLaunchDescription]:
    controller = LaunchConfiguration("controller").perform(context)
    assert controller in ["pbvs", "ibvs", "pose"]
    # Launch chosen controller
    match controller:
        case "pbvs":
            return [_include_controller_pbvs()]
        case "ibvs":
            return [_include_controller_ibvs()]
        case "pose":
            return [_include_controller_pose()]


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


def _include_controller_pose() -> IncludeLaunchDescription:
    verbose = LaunchConfiguration("verbose")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visual_control_pkg"),
                    "launch",
                    "controllers",
                    "pose_controller.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": UR_TYPE,
            "verbose": verbose,
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
        }.items(),
    )
