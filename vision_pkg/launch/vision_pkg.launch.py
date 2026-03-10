from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Isaac ROS Apriltag detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_size",
            default_value="0.08",
            description="The tag edge size in meters, assuming square markers."
            " Default value is 0.08.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_family",
            default_value="tag36h11",
            description="Tag family to detect. CUDA backend only supports tag36h11."
            " CPU and PVA backends support all choices. Default value is tag36h11.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "backends",
            default_value="CUDA",
            description="Backend to perform detection with. Default value is CUDA.",
            choices=["CUDA", "CPU", "PVA"],
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "detectors",
            default_value="apriltag",
            description="Comma separated names of nodes to launch. Default value is apriltag.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "image_topic_name",
            default_value="/image",
            description="Input image (sensor_msgs/Image) topic name to use for detector.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_info_topic_name",
            default_value="/camera_info",
            description="Camera info (sensor_msgs/CameraInfo) topic name to use for detector.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    detectors = LaunchConfiguration("detectors").perform(context)
    detectors = detectors.replace(" ", "").split(",")
    launch = []
    # Launch chosen detectors
    if "apriltag" in detectors:
        launch.append(_include_apriltag_detector())
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


def _include_apriltag_detector() -> IncludeLaunchDescription:
    tag_size = LaunchConfiguration("tag_size")
    tag_family = LaunchConfiguration("tag_family")
    backends = LaunchConfiguration("backends")

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("vision_pkg"),
                    "launch",
                    "detectors",
                    "apriltag_detector.launch.py",
                ]
            )
        ),
        launch_arguments={
            "tag_size": tag_size,
            "tag_family": tag_family,
            "backends": backends,
            "image_topic_name": image_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
        }.items(),
    )
