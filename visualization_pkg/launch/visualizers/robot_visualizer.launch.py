### Modifed launch file from the original launch file in the ur_description package.
### Original file: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/rolling/launch/view_ur.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # UR specific arguments
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
            "tf_prefix",
            default_value='""',
            description="Prefix of the joint names, useful for "
            "multi-robot setup. If changed than also joint names in the controllers' configuration "
            "have to be updated.",
        )
    )

    # Robot description arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="robot_description_pkg",
            description="Description package with robot URDF/XACRO files."
            " Defaults to robot_description_pkg.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="robot_cell.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "add_robot_cell",
            default_value="true",
            description="Add the robot cell to the robot description. Defaults to true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_isaac_cell",
            default_value="false",
            description="Use the Isaac Sim robot cell instead of the real one. Defaults to false.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "execution_mode",
            default_value="tracking",
            description="System execution mode for generating robot cell description."
            " Defaults to tracking",
            choices=["tracking", "calibration"],
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "joint_states_topic_name",
            default_value="/joint_states",
            description="Joint states (sensor_msgs/JointState) topic name to use for robot state publisher.",
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
    tf_prefix = LaunchConfiguration("tf_prefix")

    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    add_robot_cell = LaunchConfiguration("add_robot_cell")
    use_isaac_cell = LaunchConfiguration("use_isaac_cell")
    execution_mode = LaunchConfiguration("execution_mode")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")

    # Initialize robot_description parameter
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
            " ",
            "add_cell:=",
            add_robot_cell,
            " ",
            "use_isaac:=",
            use_isaac_cell,
            " ",
            "mode:=",
            execution_mode,
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "tf_prefix:=",
            tf_prefix,
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
        ]
    )
    robot_description = ParameterValue(value=robot_description_content, value_type=str)

    # Initialize nodes to start
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description}],
        remappings=[("/joint_states", joint_states_topic_name)],
    )

    nodes_to_start = [robot_state_publisher_node]
    return LaunchDescription(declared_arguments + nodes_to_start)
