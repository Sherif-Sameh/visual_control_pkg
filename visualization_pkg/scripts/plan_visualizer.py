#!/usr/bin/env python3

"""
ROS node for visualizing planned trajectories.
"""

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory
from visualization_msgs.msg import Marker


class PlanVisualizer(Node):
    """ROS node for visualizing planned trajectories in RViz."""

    def __init__(self):
        super().__init__("plan_visualizer")

        # Declare ROS parameters
        self.declare_parameter("frame.ref", rclpy.Parameter.Type.STRING)
        self.declare_parameter("traj.width", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("traj.alpha", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("traj.color", rclpy.Parameter.Type.DOUBLE_ARRAY)

        # Initialize non-ROS class attributes
        self._frame_ref = self.get_parameter("frame.ref").value
        self._traj_width = self.get_parameter("traj.width").value
        self._traj_alpha = self.get_parameter("traj.alpha").value
        self._traj_color = self.get_parameter("traj.color").value
        assert self._traj_width > 0
        assert 0 <= self._traj_alpha <= 1.0
        assert len(self._traj_color) == 3

        # Initialize ROS attributes
        self._pub_marker = self.create_publisher(Marker, "/plan_visualizer/trajectory", 1)
        self._sub_traj = self.create_subscription(
            MultiDOFJointTrajectory, "/planned_trajectory", self.callback_traj, 1
        )

    def callback_traj(self, msg: MultiDOFJointTrajectory) -> None:
        # Update trajectory
        traj = self._update_trajectory(msg)
        if not traj:
            return

        # Publish latest trajectories
        marker = Marker()
        marker.header.frame_id = self._frame_ref
        marker.header.stamp = msg.header.stamp

        marker.ns = self.get_name()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = self._traj_width
        marker.color.r = self._traj_color[0]
        marker.color.g = self._traj_color[1]
        marker.color.b = self._traj_color[2]
        marker.color.a = self._traj_alpha
        marker.points = traj
        marker.frame_locked = True
        self._pub_marker.publish(marker)

    def _update_trajectory(self, msg: MultiDOFJointTrajectory) -> list[Point]:
        traj = []
        for point in msg.points:
            translation = point.transforms[0].translation
            traj.append(Point(x=translation.x, y=translation.y, z=translation.z))
        return traj


def main(args=None):
    rclpy.init(args=args)
    plan_visualizer = PlanVisualizer()
    rclpy.spin(plan_visualizer)
    plan_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
