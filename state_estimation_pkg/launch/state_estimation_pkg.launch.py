from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Apriltag estimator arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_size",
            default_value="0.08",
            description="Tag size in meters of tracked tags. Default value is 0.08.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "estimators",
            default_value="apriltag",
            description="Comma separated names of nodes to launch. Default value is apriltag.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_info_topic_name",
            default_value="/camera_info",
            description="Camera info (sensor_msgs/CameraInfo) topic name.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_twist_topic_name",
            default_value="/camera_twist",
            description="Camera twist (geometry_msgs/TwistStamped) topic name.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="AprilTag raw detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    estimators = LaunchConfiguration("estimators").perform(context)
    estimators = estimators.replace(" ", "").split(",")
    launch = []
    # Launch chosen estimators
    if "apriltag" in estimators:
        launch.append(_include_apriltag_estimator())
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


def _include_apriltag_estimator() -> IncludeLaunchDescription:
    tag_size = LaunchConfiguration("tag_size")

    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visualization_pkg"),
                    "launch",
                    "estimators",
                    "apriltag_estimator.launch.py",
                ]
            )
        ),
        launch_arguments={
            "tag_size": tag_size,
            "camera_info_topic_name": camera_info_topic_name,
            "camera_twist_topic_name": camera_twist_topic_name,
            "detections_topic_name": detections_topic_name,
        }.items(),
    )
