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

    # Pose controller arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="Enable verbose output from pose controller. Default value is false.",
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

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_reference_topic_name",
            default_value="/goal_pose",
            description="Reference pose (geometry_msgs/PoseStamped) topic name."
            " Default is /goal_pose",
        )
    )
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

    pose_reference_topic_name = LaunchConfiguration("pose_reference_topic_name")
    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")

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
    config_path = os.path.join(pkg_share, "config", "pose_controller.toml")
    config = toml.load(config_path)

    # Initialize nodes to start
    pose_controller_node = Node(
        package="control_pkg",
        executable="pose_controller",
        output="screen",
        parameters=[
            {
                "verbose": verbose,
                "robot_description": robot_description,
                "frame.base_frame": base_frame,
                "frame.ee_frame": ee_frame,
                **config["controller"],
            }
        ],
        remappings=[
            ("/goal_pose", pose_reference_topic_name),
            ("/joint_trajectory_controller/joint_trajectory", joint_trajectory_topic_name),
            ("/joint_states", joint_states_topic_name),
        ],
    )

    nodes_to_start = [pose_controller_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
