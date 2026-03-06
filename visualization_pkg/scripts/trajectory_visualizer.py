#!/usr/bin/env python3

"""
ROS node for visualizing trajectories of selected frames from the TF tree.
"""

import math
from collections import deque

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from std_msgs.msg import Empty
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class TrajectoryVisualizer(Node):
    """ROS node for visualizing trajectories of selected frames from TF tree in RViz.

    This node listens to transforms of selected frames from the TF tree at a fixed periodic rate.
    For each transform, it maintains a fixed-length buffer of previous positions. This buffer is
    only updated if the distance between the new point and the last point in the buffer exceeds a
    set threshold. Positions are published through a visualization_msgs/MarkerArray message, where
    each entry is a line strip whose points correspond to the positions of that frame.
    """

    TIMER_PERIOD = 0.1

    def __init__(self):
        super().__init__("trajectory_visualizer")

        # Declare ROS parameters
        self.declare_parameter("frame.target", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("frame.source", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("traj.length", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("traj.spacing", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("traj.width", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("traj.alpha", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("traj.color", rclpy.Parameter.Type.DOUBLE_ARRAY)

        # Initialize non-ROS class attributes
        self._frame_target = self.get_parameter("frame.target").value
        self._frame_source = self.get_parameter("frame.source").value
        assert len(self._frame_target) == len(self._frame_source)

        traj_length = self.get_parameter("traj.length").value
        self._traj = [deque(maxlen=traj_length)] * len(self._frame_source)
        self._traj_stamp = [0.0] * len(self._frame_source)
        self._traj_spacing = self.get_parameter("traj.spacing").value
        self._traj_width = self.get_parameter("traj.width").value
        self._traj_alpha = self.get_parameter("traj.alpha").value
        self._traj_color = self.get_parameter("traj.color").value
        self._traj_color = [
            self._traj_color[3 * i : 3 * (i + 1)] for i in range(len(self._traj_color) // 3)
        ]
        assert traj_length > 0
        assert self._traj_spacing > 0
        assert self._traj_width > 0
        assert 0 <= self._traj_alpha <= 1.0
        assert len(self._traj_color) == len(self._frame_target)

        # Initialize ROS attributes
        self._pub_markers = self.create_publisher(
            MarkerArray, "/trajectory_visualizer/trajectory", 1
        )
        self._sub_rst = self.create_subscription(
            Empty, "/trajectory_visualizer/reset", self.callback_rst, 10
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._timer = self.create_timer(self.TIMER_PERIOD, self.callback_timer)

    def callback_rst(self, msg: Empty) -> None:
        """Callback function for reset message.

        Clears all stored trajectories and time stamps.
        """
        for i in range(len(self._traj)):
            self._traj[i].clear()
            self._traj_stamp[i] = 0.0

    def callback_timer(self) -> None:
        """Callback function for periodic timer.

        Updates trajectories of all tracked frames and publishes them as a
        visualization_msgs/MarkerArray message.
        """
        # Update trajectory
        self._update_trajectory()

        # Publish latest trajectories
        msg = MarkerArray()
        for i in range(len(self._traj)):
            if len(self._traj[i]) == 0:
                continue
            marker = Marker()
            marker.header.frame_id = self._frame_target[i]
            marker.header.stamp = self._traj_stamp[i]

            marker.ns = self.get_name()
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = self._traj_width
            marker.color.r = self._traj_color[i][0]
            marker.color.g = self._traj_color[i][1]
            marker.color.b = self._traj_color[i][2]
            marker.color.a = self._traj_alpha
            marker.points = list(self._traj[i])
            msg.markers.append(marker)
        self._pub_markers.publish(msg)

    def _update_trajectory(self) -> None:
        for i, (tgt_f, src_f) in enumerate(zip(self._frame_target, self._frame_source)):
            # Attempt to get latest transform
            t = self._lookup_transform(tgt_f, src_f)
            if t is None:
                continue

            translation = t.transform.translation
            pt = Point(x=translation.x, y=translation.y, z=translation.z)
            if len(self._traj[i]) == 0:
                # First time initialization
                self._traj[i].append(pt)
                self._traj_stamp[i] = t.header.stamp
            elif self._exceeds_spacing(self._traj[i][-1], pt):
                # Add latest point
                self._traj[i].append(pt)
                self._traj_stamp[i] = t.header.stamp

    def _lookup_transform(self, target_frame: str, source_frame: str) -> TransformStamped | None:
        try:
            t = self._tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return t
        except TransformException as e:
            self.get_logger().debug(f"Could not transform {source_frame} to {target_frame}: {e}")
            return None

    def _exceeds_spacing(self, old_pt: Point, new_pt: Point) -> bool:
        dx, dy, dz = new_pt.x - old_pt.x, new_pt.y - old_pt.y, new_pt.z - old_pt.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return distance > self._traj_spacing


def main(args=None):
    rclpy.init(args=args)
    trajectory_visualizer = TrajectoryVisualizer()
    rclpy.spin(trajectory_visualizer)
    trajectory_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
