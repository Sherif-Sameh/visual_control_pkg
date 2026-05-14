import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
            "pose_gt_handeye",
            default_value="[0.0]",
            description="Ground truth pose of camera wrt robot end-effector link if available."
            " To be used, the 7 pose parameters must be given in the order [tx, ty, tz, qw, qx, qy, qz]."
            " Default value is [0.0], which is internally ignored.",
        )
    )

    # TCP calibration arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_gt_tcp",
            default_value="[0.0]",
            description="Ground truth pose of TCP wrt camera if available."
            " To be used, the 7 pose parameters must be given in the order [tx, ty, tz, qw, qx, qy, qz]."
            " Default is [0.0], which is internally ignored.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "img_center",
            default_value="[240, 320]",
            description="Image center (i, j) for square crop to use for differentiable rendering."
            " Default is [240, 320].",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "modalities",
            default_value="['depth']",
            description="Rendering modalities to use for TCP pose optimization. Options include"
            " 'silhouette' and 'depth' only. Default is ['depth'].",
        )
    )

    # Eye calibration arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "marker_frame",
            default_value="tag36h11:0",
            description="Name of the reference marker frame. Default value is tag36h11:0.",
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
            "marker_id",
            default_value="0",
            description="ID of the reference marker. Default value is 0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ref_pose",
            default_value="[0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
            description="Estimate of the target eye pose wrt to the reference marker."
            " Default value is [0.075, 0.025, 0.15, 1.0, 0.0, 0.0, 0.0]",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "model",
            default_value="mipmap",
            description="Eye pose + texture model type to use. Default value is mipmap.",
            choices=["simple", "mipmap", "hashenc"],
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
            "calibration",
            default_value="tcp_calibration_p3d",
            description="Comma separated string of nodes to launch."
            " Default value is tcp_calibration_p3d.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "image_topic_name",
            default_value="/image",
            description="Input image (sensor_msgs/Image) topic name. Default is /image.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "depth_topic_name",
            default_value="/depth",
            description="Input depth (sensor_msgs/Image) topic name. Default is /depth.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_info_topic_name",
            default_value="/camera_info",
            description="Camera info (sensor_msgs/CameraInfo) topic name. Default is /camera_info.",
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
            default_value="/tcp_calibration_p3d/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /tcp_calibration_p3d/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription | Node]:
    calibration = LaunchConfiguration("calibration").perform(context)
    calibration = calibration.replace(" ", "").split(",")
    launch = []
    # Launch chosen nodes
    if "handeye_calibration" in calibration:
        launch.append(_include_handeye_calibration())
    if "handeye_evaluation" in calibration:
        launch.append(_include_handeye_evaluation())
    if (he_sb := _include_handeye_static_broadcaster(context)) is not None:
        launch.append(he_sb)
    if "tcp_calibration_p3d" in calibration:
        launch.append(_include_tcp_calibration_p3d())
    if "eye_calibration" in calibration:
        launch.append(_include_eye_calibration())
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
    pose_gt = LaunchConfiguration("pose_gt_handeye")

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


def _include_handeye_static_broadcaster(context: LaunchContext) -> Node | None:
    ee_frame = LaunchConfiguration("ee_frame").perform(context)
    cam_frame = LaunchConfiguration("cam_frame").perform(context)

    pkg_share = get_package_share_directory("calibration_pkg")
    config_path = os.path.join(
        pkg_share, "../../../../src/visual_control_pkg/calibration_pkg/config/handeye_params.toml"
    )
    if not os.path.exists(config_path):
        return None
    config = toml.load(config_path)
    pose = config["pose"]
    x, y, z = pose["translation"]
    qw, qx, qy, qz = pose["rotation"]

    # fmt: off
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="my_static_tf",
        arguments=[
            "--x", str(x), "--y", str(y), "--z", str(z),
            "--qx", str(qx), "--qy", str(qy), "--qz", str(qz), "--qw", str(qw),
            "--frame-id", ee_frame,
            "--child-frame-id", f"{cam_frame}_est",
        ],
        output="screen",
    )
    # fmt: on


def _include_tcp_calibration_p3d() -> IncludeLaunchDescription:
    pose_gt = LaunchConfiguration("pose_gt_tcp")
    img_center = LaunchConfiguration("img_center")
    modalities = LaunchConfiguration("modalities")

    image_topic_name = LaunchConfiguration("image_topic_name")
    depth_topic_name = LaunchConfiguration("depth_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("calibration_pkg"),
                    "launch",
                    "tcp",
                    "tcp_calibration_p3d.launch.py",
                ]
            )
        ),
        launch_arguments={
            "pose_gt": pose_gt,
            "img_center": img_center,
            "modalities": modalities,
            "image_topic_name": image_topic_name,
            "depth_topic_name": depth_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_eye_calibration() -> IncludeLaunchDescription:
    cam_frame = LaunchConfiguration("cam_frame")
    marker_frame = LaunchConfiguration("marker_frame")
    eye_gt_frame = LaunchConfiguration("eye_gt_frame")
    marker_id = LaunchConfiguration("marker_id")
    ref_pose = LaunchConfiguration("ref_pose")
    model = LaunchConfiguration("model")
    dr_backend = LaunchConfiguration("dr_backend")

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("calibration_pkg"), "launch", "eye", "eye_calibration.launch.py"]
            )
        ),
        launch_arguments={
            "cam_frame": cam_frame,
            "marker_frame": marker_frame,
            "eye_gt_frame": eye_gt_frame,
            "marker_id": marker_id,
            "ref_pose": ref_pose,
            "model": model,
            "dr_backend": dr_backend,
            "image_topic_name": image_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )
