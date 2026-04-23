from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # RViz arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz", default_value="true", description="Launch RViz. Defaults to true."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="ur.rviz",
            description="File name for the .rviz configuration file to load. Defaults to ur.rviz.",
        )
    )

    # Robot visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10e",
            description="Type/series of used UR robot. Default is ur10e.",
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
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_isaac_cell",
            default_value="false",
            description="Whether to use the Isaac Sim or real robot cells. Defaults to false.",
        )
    )

    # Trajectory visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "target_frames",
            default_value="['base_link']",
            description="Names of target frames for each tracked frame. Default is ['base_link'].",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "source_frames",
            default_value="['ee_link']",
            description="Names of source frames for each tracked frame. Default is ['ee_link'].",
        )
    )

    # Plan visualizer arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ref_frame",
            default_value="reference",
            description="Names of reference frame for the planned trajectory. Default is reference.",
        )
    )

    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "visualizers",
            default_value="r,t,p",
            description="Comma separated string of visualizers to enable. Use empty string to"
            " disable all. Default is 'r,t,p'.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "planned_trajectory_topic_name",
            default_value="/planned_trajectory",
            description="Planned trajectory (trajectory_msgs/MultiDOFJointTrajectory) topic name."
            " Default is /planned_trajectory.",
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
            "image_topic_name",
            default_value="/image",
            description="Image (sensor_msgs/Image) topic name. Default is /image.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "detections_topic_name",
            default_value="/detections",
            description="Detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray) topic"
            " name. Default is /detections.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/trajectory_visualizer/restart",
            description="Restart (std_msgs/Empty) topic name."
            " Default is /trajectory_visualizer/restart",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[Node | IncludeLaunchDescription]:
    rviz = LaunchConfiguration("rviz").perform(context)
    visualizers = LaunchConfiguration("visualizers").perform(context)
    visualizers = visualizers.replace(" ", "").split(",")
    launch = []
    # Launch RViz if required
    if rviz == "true":
        launch.append(_include_rviz_node())
    # Launch chosen visualizers
    if "r" in visualizers:
        launch.append(_include_robot_visualizer())
    if "t" in visualizers:
        launch.append(_include_trajectory_visualizer())
    if "d" in visualizers:
        launch.append(_include_detection_visualizer())
    if "p" in visualizers:
        launch.append(_include_plan_visualizer())
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


def _include_rviz_node() -> Node:
    rviz_config = LaunchConfiguration("rviz_config")
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("visualization_pkg"), "rviz", rviz_config]
    )
    return Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )


def _include_robot_visualizer() -> IncludeLaunchDescription:
    ur_type = LaunchConfiguration("ur_type")
    execution_mode = LaunchConfiguration("execution_mode")
    use_isaac_cell = LaunchConfiguration("use_isaac_cell")

    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visualization_pkg"),
                    "launch",
                    "visualizers",
                    "robot_visualizer.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "execution_mode": execution_mode,
            "use_isaac_cell": use_isaac_cell,
            "joint_states_topic_name": joint_states_topic_name,
        }.items(),
    )


def _include_trajectory_visualizer() -> IncludeLaunchDescription:
    target_frames = LaunchConfiguration("target_frames")
    source_frames = LaunchConfiguration("source_frames")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visualization_pkg"),
                    "launch",
                    "visualizers",
                    "trajectory_visualizer.launch.py",
                ]
            )
        ),
        launch_arguments={
            "target_frames": target_frames,
            "source_frames": source_frames,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )


def _include_detection_visualizer() -> IncludeLaunchDescription:
    image_topic_name = LaunchConfiguration("image_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visualization_pkg"),
                    "launch",
                    "visualizers",
                    "detection_visualizer.launch.py",
                ]
            )
        ),
        launch_arguments={
            "image_topic_name": image_topic_name,
            "detections_topic_name": detections_topic_name,
        }.items(),
    )


def _include_plan_visualizer() -> IncludeLaunchDescription:
    ref_frame = LaunchConfiguration("ref_frame")
    planned_trajectory_topic_name = LaunchConfiguration("planned_trajectory_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("visualization_pkg"),
                    "launch",
                    "visualizers",
                    "plan_visualizer.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ref": ref_frame,
            "planned_trajectory_topic_name": planned_trajectory_topic_name,
        }.items(),
    )
