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

    # ChArUco detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualize",
            default_value="false",
            description="Enable ChArUco detection visualization topic. Default value is false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "dict_name",
            default_value="DICT_5X5_50",
            description="Name of the predefined marker dictionary. Default value is DICT_5X5_50.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_xs",
            default_value="10",
            description="Number of chessboard squares in X direction. Default value is 10.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_ys",
            default_value="10",
            description="Number of chessboard squares in Y direction. Default value is 10.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_sq_len",
            default_value="0.04",
            description="Chessboard square side length in meters. Default value is 0.04.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "board_mk_len",
            default_value="0.03",
            description="Marker side length in meters. Default value is 0.03.",
        )
    )

    # Eye detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "marker_frame",
            default_value="tag36h11:0",
            description="Name of the reference marker frame for eye detector. Default value is tag36h11:0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "eye_gt_frame",
            default_value="eye_left",
            description="Name of the ground truth target eye frame. Default value is eye_left.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ref_pose",
            default_value="[0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
            description="Estimate of the eye pose wrt to the reference marker."
            " Default value is [0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "dr_backend",
            default_value="cuda",
            description="Kaolin differentiable rendering backend. Default value is cuda.",
            choices=["cuda", "nvdiffrast"],
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "detector",
            default_value="apriltag",
            description="Comma separated string of detector nodes to launch. Default value is apriltag.",
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
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/eye_detector/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /eye_detector/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    detector = LaunchConfiguration("detector").perform(context)
    detector = detector.replace(" ", "").split(",")
    launch = []
    # Launch chosen detectors
    if "apriltag" in detector:
        launch.append(_include_apriltag_detector())
    if "charuco" in detector:
        launch.append(_include_charuco_detector())
    if "eye" in detector:
        launch.append(_include_eye_detector())
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


def _include_charuco_detector() -> IncludeLaunchDescription:
    visualize = LaunchConfiguration("visualize")
    dict_name = LaunchConfiguration("dict_name")
    board_xs = LaunchConfiguration("board_xs")
    board_ys = LaunchConfiguration("board_ys")
    board_sq_len = LaunchConfiguration("board_sq_len")
    board_mk_len = LaunchConfiguration("board_mk_len")

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("vision_pkg"),
                    "launch",
                    "detectors",
                    "charuco_detector.launch.py",
                ]
            )
        ),
        launch_arguments={
            "visualize": visualize,
            "dict_name": dict_name,
            "board_xs": board_xs,
            "board_ys": board_ys,
            "board_sq_len": board_sq_len,
            "board_mk_len": board_mk_len,
            "image_topic_name": image_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
        }.items(),
    )


def _include_eye_detector() -> IncludeLaunchDescription:
    marker_frame = LaunchConfiguration("marker_frame")
    eye_gt_frame = LaunchConfiguration("eye_gt_frame")
    ref_pose = LaunchConfiguration("ref_pose")
    dr_backend = LaunchConfiguration("dr_backend")

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("vision_pkg"), "launch", "detectors", "eye_detector.launch.py"]
            )
        ),
        launch_arguments={
            "marker_frame": marker_frame,
            "eye_gt_frame": eye_gt_frame,
            "ref_pose": ref_pose,
            "dr_backend": dr_backend,
            "image_topic_name": image_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )
