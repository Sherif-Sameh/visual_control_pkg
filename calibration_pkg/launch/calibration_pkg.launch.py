from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Hand-eye calibration/evaluation arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Name of the base frame of the robot. Default value is base_link.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ee_frame",
            default_value="ee_link",
            description="Name of the end-effector frame of the robot. Default value is ee_link.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "cam_frame",
            default_value="camera_color_optical_frame",
            description="Name of the camera frame of the RGB images."
            " Default value is camera_color_optical_frame.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_gt",
            default_value="[0.0]",
            description="Ground truth pose of camera wrt robot end-effector link if available."
            " To be used, the 7 pose parameters must be given in the order [tx, ty, tz, qw, qx, qy, qz]."
            " Default value is [0.0], which is internally ignored.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "calibration",
            default_value="handeye_calibration",
            description="Comma separated string of nodes to launch."
            " Default value is handeye_calibration.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="AprilTag detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/handeye_calibration/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /handeye_calibration/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    calibration = LaunchConfiguration("calibration").perform(context)
    calibration = calibration.replace(" ", "").split(",")
    launch = []
    # Launch chosen nodes
    if "handeye_calibration" in calibration:
        launch.append(_include_handeye_calibration())
    if "handeye_evaluation" in calibration:
        launch.append(_include_handeye_evaluation())
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


def _include_handeye_calibration() -> IncludeLaunchDescription:
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")
    cam_frame = LaunchConfiguration("cam_frame")
    pose_gt = LaunchConfiguration("pose_gt")

    detections_topic_name = LaunchConfiguration("detections_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("calibration_pkg"),
                    "launch",
                    "handeye",
                    "handeye_calibration.launch.py",
                ]
            )
        ),
        launch_arguments={
            "base_frame": base_frame,
            "ee_frame": ee_frame,
            "cam_frame": cam_frame,
            "pose_gt": pose_gt,
            "detections_topic_name": detections_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_handeye_evaluation() -> IncludeLaunchDescription:
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")

    detections_topic_name = LaunchConfiguration("detections_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("calibration_pkg"),
                    "launch",
                    "handeye",
                    "handeye_evaluation.launch.py",
                ]
            )
        ),
        launch_arguments={
            "base_frame": base_frame,
            "ee_frame": ee_frame,
            "detections_topic_name": detections_topic_name,
        }.items(),
    )
