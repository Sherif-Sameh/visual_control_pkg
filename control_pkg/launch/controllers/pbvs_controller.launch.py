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
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur_description",
            description="Description package with robot URDF/XACRO files. Usually the argument "
            "is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="ur.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
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
            "tag_family",
            default_value="tag36h11",
            description="Tag family to use for tracking. Default value is tag36h11.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_ids",
            default_value="[0]",
            description="Tag IDS to use for tracking. Default value is [0].",
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
    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    tf_prefix = LaunchConfiguration("tf_prefix")

    verbose = LaunchConfiguration("verbose")
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")
    cam_frame = LaunchConfiguration("cam_frame")
    tag_family = LaunchConfiguration("tag_family")
    tag_ids = LaunchConfiguration("tag_ids")

    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    # Initialize robot_description parameter
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            "ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "tf_prefix:=",
            tf_prefix,
        ]
    )
    robot_description = ParameterValue(value=robot_description_content, value_type=str)

    # Load configuration from toml
    pkg_share = get_package_share_directory("control_pkg")
    config_path = os.path.join(pkg_share, "config", "pbvs_controller.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    pbvs_controller_node = Node(
        package="control_pkg",
        executable="pbvs_controller",
        output="screen",
        parameters=[
            {
                "verbose": verbose,
                "robot_description": robot_description,
                "frame.base_frame": base_frame,
                "frame.ee_frame": ee_frame,
                "frame.cam_frame": cam_frame,
                "tag.tag_family": tag_family,
                "tag.tag_ids": tag_ids,
                **config["controller"],
            }
        ],
        remappings=[
            ("/joint_trajectory_controller/joint_trajectory", joint_trajectory_topic_name),
            ("/joint_states", joint_states_topic_name),
            ("/detections", detections_topic_name),
        ],
    )

    nodes_to_start = [pbvs_controller_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
