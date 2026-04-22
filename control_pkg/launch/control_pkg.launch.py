from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Pose/PBVS/IBVS controllers common arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10e",
            description="Type/series of used UR robot. Default is ur10e.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="Enable verbose output from controller. Default value is false.",
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

    # PBVS/IBVS controllers common arguments
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
            "tag_id",
            default_value="0",
            description="Tag ID to use for tracking. Default value is 0.",
        )
    )

    # IBVS controller unique arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_size",
            default_value="0.08",
            description="Tag size in meters to use for tracking. Default value is 0.08.",
        )
    )

    # OC planner unique arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "tcp_frame",
            default_value="tcp",
            description="Name of the tcp frame of the robot. Default value is tcp.",
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
            "controller",
            default_value="pbvs",
            description="Controller to use for tracking. Default is pbvs",
            choices=["pbvs", "ibvs", "pose"],
        )
    )
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
            description="Detections (isaac_ros_apriltag_interfaces/AprilTagDetectionArray) topic"
            " name. Default is /detections.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "restart_topic_name",
            default_value="/oc_planner/restart",
            description="Restart (std_msgs/Empty) topic name. Default is /oc_planner/restart.",
        )
    )
    return declared_arguments


def launch_setup(context: LaunchContext) -> list[IncludeLaunchDescription]:
    controller = LaunchConfiguration("controller").perform(context)
    assert controller in ["pbvs", "ibvs", "pose"]
    # Launch chosen controller
    nodes = []
    match controller:
        case "pbvs":
            nodes.append(_include_pbvs_controller())
        case "ibvs":
            nodes.append(_include_ibvs_controller())
        case "pose":
            nodes.append(_include_pose_controller())
    # Launch OC planner if needed
    if controller in ["pbvs", "ibvs"]:
        nodes.append(_include_oc_planner(controller))
    return nodes


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Add opaque functions
    opaque_functions = [OpaqueFunction(function=launch_setup)]
    return LaunchDescription(declared_arguments + opaque_functions)


##
# Private functions
##


def _include_pbvs_controller() -> IncludeLaunchDescription:
    ur_type = LaunchConfiguration("ur_type")
    verbose = LaunchConfiguration("verbose")
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")
    cam_frame = LaunchConfiguration("cam_frame")
    tag_id = LaunchConfiguration("tag_id")

    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")
    desired_trajectory_topic_name = "/oc_planner/trajectory"

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("control_pkg"),
                    "launch",
                    "controllers",
                    "pbvs_controller.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "verbose": verbose,
            "base_frame": base_frame,
            "ee_frame": ee_frame,
            "cam_frame": cam_frame,
            "tag_id": tag_id,
            "joint_trajectory_topic_name": joint_trajectory_topic_name,
            "joint_states_topic_name": joint_states_topic_name,
            "detections_topic_name": detections_topic_name,
            "desired_trajectory_topic_name": desired_trajectory_topic_name,
        }.items(),
    )


def _include_ibvs_controller() -> IncludeLaunchDescription:
    ur_type = LaunchConfiguration("ur_type")
    verbose = LaunchConfiguration("verbose")
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")
    cam_frame = LaunchConfiguration("cam_frame")
    tag_size = LaunchConfiguration("tag_size")
    tag_ids = LaunchConfiguration("tag_ids")

    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    detections_topic_name = LaunchConfiguration("detections_topic_name")
    desired_trajectory_topic_name = "/oc_planner/trajectory"

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("control_pkg"),
                    "launch",
                    "controllers",
                    "ibvs_controller.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "verbose": verbose,
            "base_frame": base_frame,
            "ee_frame": ee_frame,
            "cam_frame": cam_frame,
            "tag_size": tag_size,
            "tag_ids": tag_ids,
            "joint_trajectory_topic_name": joint_trajectory_topic_name,
            "joint_states_topic_name": joint_states_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
            "detections_topic_name": detections_topic_name,
            "desired_trajectory_topic_name": desired_trajectory_topic_name,
        }.items(),
    )


def _include_pose_controller() -> IncludeLaunchDescription:
    ur_type = LaunchConfiguration("ur_type")
    verbose = LaunchConfiguration("verbose")
    base_frame = LaunchConfiguration("base_frame")
    ee_frame = LaunchConfiguration("ee_frame")

    joint_trajectory_topic_name = LaunchConfiguration("joint_trajectory_topic_name")
    joint_states_topic_name = LaunchConfiguration("joint_states_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("control_pkg"),
                    "launch",
                    "controllers",
                    "pose_controller.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "verbose": verbose,
            "base_frame": base_frame,
            "ee_frame": ee_frame,
            "joint_trajectory_topic_name": joint_trajectory_topic_name,
            "joint_states_topic_name": joint_states_topic_name,
        }.items(),
    )


def _include_oc_planner(controller: str) -> IncludeLaunchDescription:
    cam_frame = LaunchConfiguration("cam_frame")
    tcp_frame = LaunchConfiguration("tcp_frame")
    pose_mk_tgt = LaunchConfiguration("pose_mk_tgt")

    state_reference_topic_name = LaunchConfiguration("state_reference_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")
    camera_twist_topic_name = f"{controller}_controller/camera_twist"
    detections_topic_name = LaunchConfiguration("detections_topic_name")
    restart_topic_name = LaunchConfiguration("restart_topic_name")

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("control_pkg"), "launch", "planners", "oc_planner.launch.py"]
            )
        ),
        launch_arguments={
            "cam_frame": cam_frame,
            "tcp_frame": tcp_frame,
            "pose_mk_gt": pose_mk_tgt,
            "state_reference_topic_name": state_reference_topic_name,
            "camera_info_topic_name": camera_info_topic_name,
            "camera_twist_topic_name": camera_twist_topic_name,
            "detections_topic_name": detections_topic_name,
            "restart_topic_name": restart_topic_name,
        }.items(),
    )
