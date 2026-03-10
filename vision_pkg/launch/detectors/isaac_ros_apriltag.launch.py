from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def declare_arguments() -> list[DeclareLaunchArgument]:
    declared_arguments = []

    # Isaac ROS Apriltag detector arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "size",
            default_value="0.08",
            description="The tag edge size in meters, assuming square markers."
            " Default value is 0.08.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "max_tags",
            default_value="16",
            description="The maximum number of tags to be detected. Default value is 16.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tile_size",
            default_value="4",
            description="Tile/window size used for adaptive thresholding in pixels."
            " Default value is 4.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tag_family",
            default_value="tag36h11",
            description="Tag family to detect. CUDA backend only supports tag36h11."
            " CPU and PVA backends support all choices. Default value is tag36h11.",
            choices=[
                "tag36h11",
                "tag16h5",
                "tag25h9",
                "tag36h10",
                "tag36h11",
                "circle21h7",
                "circle49h12",
                "custom48h12",
                "standard41h12",
                "standard52h13",
            ],
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

    # General arguments
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
    return declared_arguments


def get_composable_node(**kwargs) -> ComposableNode:
    # Initialize Arguments
    size = LaunchConfiguration("size")
    max_tags = LaunchConfiguration("max_tags")
    tile_size = LaunchConfiguration("tile_size")
    tag_family = LaunchConfiguration("tag_family")
    backends = LaunchConfiguration("backends")

    image_topic_name = LaunchConfiguration("image_topic_name")
    camera_info_topic_name = LaunchConfiguration("camera_info_topic_name")

    # Initialize composable node
    return ComposableNode(
        package="isaac_ros_apriltag",
        plugin="nvidia::isaac_ros::apriltag::AprilTagNode",
        name="apriltag",
        parameters=[
            {
                "size": kwargs.get("size", size),
                "max_tags": kwargs.get("max_tags", max_tags),
                "tile_size": kwargs.get("tile_size", tile_size),
                "tag_family": kwargs.get("tag_family", tag_family),
                "backends": kwargs.get("backends", backends),
            }
        ],
        remappings=[
            ("/image", kwargs.get("image_topic_name", image_topic_name)),
            ("/camera_info", kwargs.get("camera_info_topic_name", camera_info_topic_name)),
            ("/tag_detections", "/detector/tag_detections"),
        ],
    )


def generate_launch_description() -> LaunchDescription:
    # Declare arguments
    declared_arguments = declare_arguments()

    # Initialize composable node
    apriltag_node = get_composable_node()

    # Initialize standalone composable node container
    apriltag_detector_container = ComposableNodeContainer(
        package="rclcpp_components",
        name="apriltag_detector_container",
        namespace="",
        executable="component_container_mt",
        composable_node_descriptions=[apriltag_node],
        output="screen",
    )

    nodes_to_start = [apriltag_detector_container]
    return LaunchDescription(declared_arguments + nodes_to_start)
