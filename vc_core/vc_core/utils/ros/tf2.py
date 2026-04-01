import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer


def lookup_transform(
    target_frame: str, source_frame: str, buffer: Buffer
) -> TransformStamped | None:
    """Lookup transform from TF tree between target and source frames.

    Args:
        target_frame: Target frame for desired transform.
        source_frame: Source frame for desired transform.
        buffer: TF2 buffer to use for frame lookup.

    Returns:
        Optional transform as a `geometry_msgs.msg.TransformStamped` if lookup was successful or
        `None` otherwise if a `tf2_ros.TransformException` was raised.
    """
    try:
        return buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
    except TransformException:
        return None
