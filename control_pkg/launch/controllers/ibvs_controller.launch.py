import os

import toml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # UR description-related arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
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
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            default_value='""',
            description="Prefix of the joint names, useful for "
            "multi-robot setup. If changed than also joint names in the controllers' configuration "
            "have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "kinematics_params",
            default_value="unused",
            description="Config file containing the calibration values extracted from the robot.",
        )
    )

    # IBVS controller arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="Enable verbose output from IBVS controller. Default value is false.",
        )
    )
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
            "tag_size",
            default_value="0.08",
            description="Tag size in meters to use for tracking. Default value is 0.08.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_id",
            default_value="0",
            description="Tag ID to use for tracking. Default value is 0.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "joint_trajectory_topic_name",
            default_value="/joint_trajectory_controller/joint_trajectory",
            description="Joint trajectory (trajectory_msgs/JointTrajectory) topic name."
            " Default is /joint_trajectory_controller/joint_trajectory.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "joint_states_topic_name",
            default_value="/joint_states",
            description="Joint states (sensor_msgs/JointState) topic name."
            " Default is /joint_states.",
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
            "detections_topic_name",
            default_value="/detections",
            description="AprilTag detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray)"
            " topic name. Default is /detections.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "desired_trajectory_topic_name",
            default_value="/desired_trajectory",
            description="Desired trajectory (trajectory_msgs/MultiDOFJointTrajectory) topic name."
            " Default is /desired_trajectory.",
        )
    )
    return declared_arguments


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    tf_prefix = LaunchConfiguration("tf_prefix")

    verbose = LaunchConfiguration("verbose")
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")
    cam_frame = LaunchConfiguration("cam_frame")
    tag_size = LaunchConfiguration("tag_size")
    tag_id = LaunchConfiguration("tag_id")

    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")
    desired_trajectory_topic_name = LaunchConfiguration("desired_trajectory_topic_name")

    # Initialize robot_description parameter
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]),
            " ",
            "name:=ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "tf_prefix:=",
            tf_prefix,
            " ",
            "safety_limits:=true",
            " ",
            "safety_pos_margin:=0.15",
            " ",
            "safety_k_position:=20",
        ]
    )
    robot_description = ParameterValue(value=robot_description_content, value_type=str)

    # Load configuration from toml
    pkg_share = get_package_share_directory("control_pkg")
    config_path = os.path.join(pkg_share, "config", "ibvs_controller.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    ibvs_controller_node = Node(
        package="control_pkg",
        executable="ibvs_controller",
        output="screen",
        parameters=[
            {
                "verbose": verbose,
                "robot_description": robot_description,
                "frame.base_frame": base_frame,
                "frame.ee_frame": ee_frame,
                "frame.cam_frame": cam_frame,
                "tag.tag_size": tag_size,
                "tag.tag_id": tag_id,
                **config["controller"],
            }
        ],
        remappings=[
            ("/joint_trajectory_controller/joint_trajectory", joint_trajectory_topic_name),
            ("/joint_states", joint_states_topic_name),
            ("/camera_info", camera_info_topic_name),
            ("/detections", detections_topic_name),
            ("/desired_trajectory", desired_trajectory_topic_name),
        ],
    )

    nodes_to_start = [ibvs_controller_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
