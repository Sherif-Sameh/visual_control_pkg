import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # OC planner arguments
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
            "tcp_frame",
            default_value="tcp",
            description="Name of the tcp frame of the robot. Default value is tcp.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_id",
            default_value="0",
            description="Tag ID to use for tracking. Default value is 0.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_mk_tgt",
            default_value="[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
            description="Pose of target wrt the reference marker."
            " Default value is [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "state_reference_topic_name",
            default_value="/state_reference",
            description="Reference state (trajectory_msgs/MultiDOFJointTrajectory) topic name."
            " Default is /state_reference",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_twist_topic_name",
            default_value="/camera_twist",
            description="Camera twist (geometry_msgs/TwistStamped) topic name."
            " Default is /camera_twist.",
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
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    cam_frame = LaunchConfiguration("cam_frame")
    tcp_frame = LaunchConfiguration("tcp_frame")
    tag_id = LaunchConfiguration("tag_id")
    pose_mk_tgt = LaunchConfiguration("pose_mk_tgt")

    state_reference_topic_name = LaunchConfiguration("state_reference_topic_name")
    camera_twist_topic_name = LaunchConfiguration("camera_twist_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Load configuration from toml
    pkg_share = get_package_share_directory("control_pkg")
    config_path = os.path.join(pkg_share, "config", "oc_planner.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    oc_planner_node = Node(
        package="control_pkg",
        executable="oc_planner.py",
        output="screen",
        parameters=[
            {
                "frame.cam": cam_frame,
                "frame.tcp": tcp_frame,
                "pose.mk_tgt": pose_mk_tgt,
                "tag.tag_id": tag_id,
                **config["planner"],
            }
        ],
        remappings=[
            ("/state_reference", state_reference_topic_name),
            ("/camera_twist", camera_twist_topic_name),
            ("/detections", detections_topic_name),
        ],
    )

    nodes_to_start = [oc_planner_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
