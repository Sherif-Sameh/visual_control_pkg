from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Apriltag/ChArUco estimator arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "marker_size",
            default_value="0.08",
            description="Marker size in meters. Default value is 0.08.",
        )
    )

    # Pose estimator arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_frame",
            default_value="pose",
            description="Name of the estimated pose frame. Default value is pose.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "estimator",
            default_value="apriltag",
            description="Comma separated string of estimator nodes to launch. Default value is apriltag.",
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
            description="Raw detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_topic_name",
            default_value="/pose",
            description="Pose raw measurements (isaac_ros_apriltag_interfaces/AprilTagDetectionArray) topic name."
            " Default is /pose.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/pose_estimator/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /pose_estimator/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    estimator = LaunchConfiguration("estimator").perform(context)
    estimator = estimator.replace(" ", "").split(",")
    launch = []
    # Launch chosen estimator
    if "apriltag" in estimator:
        launch.append(_include_apriltag_estimator())
    if "charuco" in estimator:
        launch.append(_include_charuco_estimator())
    if "pose" in estimator:
        launch.append(_include_pose_estimator())
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
    tag_size = LaunchConfiguration("marker_size")

    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("state_estimation_pkg"),
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
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_charuco_estimator() -> IncludeLaunchDescription:
    board_size = LaunchConfiguration("marker_size")

    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("state_estimation_pkg"),
                    "launch",
                    "estimators",
                    "charuco_estimator.launch.py",
                ]
            )
        ),
        launch_arguments={
            "board_size": board_size,
            "camera_info_topic_name": camera_info_topic_name,
            "camera_twist_topic_name": camera_twist_topic_name,
            "detections_topic_name": detections_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_pose_estimator() -> IncludeLaunchDescription:
    pose_frame = LaunchConfiguration("pose_frame")

    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    pose_topic_name = LaunchConfiguration("pose_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("state_estimation_pkg"),
                    "launch",
                    "estimators",
                    "pose_estimator.launch.py",
                ]
            )
        ),
        launch_arguments={
            "pose_frame": pose_frame,
            "camera_twist_topic_name": camera_twist_topic_name,
            "pose_topic_name": pose_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )
