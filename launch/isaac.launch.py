from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10e",
            description="Type/series of used UR robot.",
            choices=[
                "ur3",
                "ur5",
                "ur10",
                "ur3e",
                "ur5e",
                "ur7e",
                "ur10e",
                "ur12e",
                "ur16e",
                "ur8long",
                "ur15",
                "ur18",
                "ur20",
                "ur30",
            ],
        )
    )

    # Isaac ROS Apriltag detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "size",
            default_value="0.08",
            description="The tag edge size in meters, assuming square markers."
            " Default value is 0.08.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "backends",
            default_value="CPU",
            description="Backend to perform detection with. Default value is CPU.",
            choices=[
                "CUDA",
                "CPU",
                "PVA",
            ],
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

    # PBVS controller arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="Enable verbose output from PBVS controller. Default value is false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_family",
            default_value="tag36h11",
            description="Tag family to use for tracking. Default value is tag36h11.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_ids",
            default_value="[0, 1]",
            description="Tag IDS to use for tracking. Default value is [0, 1].",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="isaac.rviz",
            description="File name for the .rviz configuration file to load."
            " Defaults to isaac.rviz.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")

    size = LaunchConfiguration("size")
    backends = LaunchConfiguration("backends")
    use_debugger = LaunchConfiguration("use_debugger")

    verbose = LaunchConfiguration("verbose")
    BASE_FRAME = "base_link"
    EE_FRAME = "wrist_3_link"
    CAM_FRAME = "camera_color_optical_frame"
    tag_ids = LaunchConfiguration("tag_ids")
    tag_family = LaunchConfiguration("tag_family")

    rviz_config = LaunchConfiguration("rviz_config")
    CAMERA_INFO_TOPIC_NAME = "/isaaclab/camera/camera_info"
    DETECTIONS_TOPIC_NAME = "/detector/tag_detections"
    IMAGE_TOPIC_NAME = "/isaaclab/camera/image_raw"
    JOINT_STATES_TOPIC_NAME = "/isaaclab/joint_states"
    JOINT_TRAJECTORY_TOPIC_NAME = (
        "/isaaclab/joint_trajectory_controller/joint_trajectory"
    )

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
            "ur_type": ur_type,
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
            "size": size,
            "backends": backends,
            "use_debugger": use_debugger,
            "image_topic_name": IMAGE_TOPIC_NAME,
            "camera_info_topic_name": CAMERA_INFO_TOPIC_NAME,
        }.items(),
    )

    pbvs_controller_launch = IncludeLaunchDescription(
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
            "ur_type": ur_type,
            "verbose": verbose,
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "cam_frame": CAM_FRAME,
            "tag_family": tag_family,
            "tag_ids": tag_ids,
            "joint_trajectory_topic_name": JOINT_TRAJECTORY_TOPIC_NAME,
            "joint_states_topic_name": JOINT_STATES_TOPIC_NAME,
            "detections_topic_name": DETECTIONS_TOPIC_NAME,
        }.items(),
    )

    launch_files = [
        view_ur_launch,
        isaac_ros_apriltag_launch,
        pbvs_controller_launch,
    ]
    return LaunchDescription(declared_arguments + launch_files)
