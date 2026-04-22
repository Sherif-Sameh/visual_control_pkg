#!/usr/bin/env python3

"""
ROS node for optimal control-based trajectory planning using Acados
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Transform, Twist, TwistStamped
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from numpy.typing import NDArray
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Empty, Float64
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from vc_core.ocp.solver import VsOcpSolver, VsOcpSolverCfg
from vc_core.utils.ros.tf2 import lookup_transform


class OcPlanner(Node):
    """ROS node for optimal control-based trajectory planning using Acados."""

    def __init__(self):
        super().__init__("oc_planner")

        # Declare ROS parameters
        self.declare_parameter("n_update", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("frame.cam", rclpy.Parameter.Type.STRING)
        self.declare_parameter("frame.tcp", rclpy.Parameter.Type.STRING)
        self.declare_parameter("pose.mk_tgt", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("model.alpha", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("model.fp", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("cost.Q_x_diag", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("cost.R_u_diag", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("cost.Q_z_diag", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.lbx", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.ubx", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.idxbx", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.lbu", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.ubu", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.idxbu", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.lh", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.uh", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("solver.n_horizon", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("solver.time_step", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("solver.nlp_solver_max_iter", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("solver.nlp_tol", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("solver.qp_solver_iter_max", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("solver.qp_tol", rclpy.Parameter.Type.DOUBLE)

        # Initialize non-ROS class attributes
        pose_mk_tgt = self.get_parameter("pose.mk_tgt").value
        self._n_update = self.get_parameter("n_update").value
        self._time_step = self.get_parameter("solver.time_step").value
        self._traj_stamp = self.get_clock().now()
        self._pose_mk_tgt = (
            np.array(pose_mk_tgt[:3]),
            R.from_quat(pose_mk_tgt[3:], scalar_first=True),
        )
        self._pose_tcp_cam = None
        self._reset()

        # Initialize ROS attributes
        self._pub_traj = self.create_publisher(MultiDOFJointTrajectory, "/oc_planner/trajectory", 0)
        self._pub_exec_time = self.create_publisher(Float64, "/oc_planner/execution_time", 0)
        self._sub_pref = self.create_subscription(
            PoseStamped, "/pose_reference", self.callback_pref, 0
        )
        self._sub_cam_info = self.create_subscription(
            CameraInfo, "/camera_info", self.callback_cam_info, 0
        )
        self._sub_cam_twist = self.create_subscription(
            TwistStamped, "/camera_twist", self.callback_cam_twist, 0
        )
        self._sub_dtn = self.create_subscription(
            AprilTagDetectionArray, "/detections", self.callback_dtn, 0
        )
        self._sub_rst = self.create_subscription(Empty, "/oc_planner/restart", self.callback_rst, 0)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def publish_traj(self, id: int, pose: NDArray, twist: NDArray) -> None:
        """Publish the planned trajectory."""
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names.append(str(id))

        traj_len = min(2 * self._n_update, pose.shape[0])
        for i in range(traj_len):
            t = Transform()
            t.translation.x = float(pose[i, 0])
            t.translation.y = float(pose[i, 1])
            t.translation.z = float(pose[i, 2])
            t.rotation.w = float(pose[i, 3])
            t.rotation.x = float(pose[i, 4])
            t.rotation.y = float(pose[i, 5])
            t.rotation.z = float(pose[i, 6])
            tw = Twist()
            tw.linear.x = float(twist[i, 0])
            tw.linear.y = float(twist[i, 1])
            tw.linear.z = float(twist[i, 2])
            tw.angular.x = float(twist[i, 3])
            tw.angular.y = float(twist[i, 4])
            tw.angular.z = float(twist[i, 5])
            msg.points.append(
                MultiDOFJointTrajectoryPoint(
                    transforms=[t],
                    velocities=[tw],
                    time_from_start=Duration(seconds=i * self._time_step).to_msg(),
                )
            )
        self._pub_traj.publish(msg)

    def publish_exec_time(self, time: float) -> None:
        """Publish the execution time taken by the solver."""
        msg = Float64(data=time)
        self._pub_exec_time.publish(msg)

    def callback_pref(self, msg: PoseStamped) -> None:
        pos, ori = msg.pose.position, msg.pose.orientation
        tvec = np.array([pos.x, pos.y, pos.z])
        rot = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        self._pose_tgt_tcpd = (tvec, rot)

    def callback_cam_info(self, msg: CameraInfo) -> None:
        if self._ocp_solver is None:
            fxy = np.array([msg.k[0], msg.k[4]], dtype=np.float64)
            self._ocp_solver = self._init_ocp_solver(fxy)

    def callback_cam_twist(self, msg: TwistStamped) -> None:
        self._twist[:3] = msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
        self._twist[3:] = msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z

    def callback_dtn(self, msg: AprilTagDetectionArray) -> None:
        # resolve camera wrt TCP transform
        if self._pose_tcp_cam is None:
            self._pose_tcp_cam = self._lookup_transform()
        no_ref = self._pose_tgt_tcpd is None and self._pose_mk_camd is None
        if no_ref or self._ocp_solver is None or self._pose_tcp_cam is None:
            return
        elapsed = self._get_elapsed()
        if self._pose_tgt_tcpd is None and elapsed < self._n_update * self._time_step:
            return
        # compute camera pose wrt marker
        pose = msg.detections[0].pose.pose.pose
        pos, ori = pose.position, pose.orientation
        rot_mk_cam = R.from_quat([ori.x, ori.y, ori.z, ori.w]).inv()
        tvec_mk_cam = -rot_mk_cam.apply([pos.x, pos.y, pos.z])
        pose_mk_cam = np.concatenate([tvec_mk_cam, rot_mk_cam.as_quat(scalar_first=True)])
        # update reference if a new reference is available
        if self._pose_tgt_tcpd is not None:
            pose_tgt_camd = self._pose_mult(self._pose_tgt_tcpd, self._pose_tcp_cam)
            self._pose_mk_camd = self._pose_mult(self._pose_mk_tgt, pose_tgt_camd)
        # get and publish new trajectory
        pose, twist = self._solve_for_trajectory(pose_mk_cam, elapsed)
        self._pose_tgt_tcpd = None
        self.publish_traj(msg.detections[0].id, pose, twist)
        self.publish_exec_time(self._ocp_solver.get_stats("time_tot"))
        self._traj_stamp = self.get_clock().now()

    def callback_rst(self, msg: Empty) -> None:
        self._reset()

    def _init_ocp_solver(self, fxy: NDArray) -> VsOcpSolver:
        """Initialize the VS OCP solver."""
        model_params = {k: v.value for k, v in self.get_parameters_by_prefix("model").items()}
        cost_params = {k: v.value for k, v in self.get_parameters_by_prefix("cost").items()}
        constraint_params = {
            k: np.array(v.value) for k, v in self.get_parameters_by_prefix("constraint").items()
        }
        constraint_params["lh"] /= fxy
        constraint_params["uh"] /= fxy
        solver_params = {k: v.value for k, v in self.get_parameters_by_prefix("solver").items()}

        ocp_cfg = VsOcpSolverCfg(
            alpha=model_params["alpha"],
            fp=np.array(model_params["fp"]),
            cost_cfg=VsOcpSolverCfg.CostCfg(
                Q_x=np.diag(cost_params["Q_x_diag"]),
                R_u=np.diag(cost_params["R_u_diag"]),
                Q_z=np.diag(cost_params["Q_z_diag"]),
            ),
            constraint_cfg=VsOcpSolverCfg.ConstraintCfg(**constraint_params),
            solver_cfg=VsOcpSolverCfg.SolverCfg(**solver_params),
        )
        return VsOcpSolver(ocp_cfg)

    def _lookup_transform(self) -> tuple[NDArray, R] | None:
        """Lookup camera wrt TCP transform."""
        cam_frame = self.get_parameter("frame.cam").value
        tcp_frame = self.get_parameter("frame.tcp").value
        t = lookup_transform(tcp_frame, cam_frame, self._tf_buffer)
        if t is None:
            return None
        pos, ori = t.transform.translation, t.transform.rotation
        tvec = np.array([pos.x, pos.y, pos.z])
        rot = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        return tvec, rot

    @staticmethod
    def _pose_mult(pose01: tuple[NDArray, R], pose12: tuple[NDArray, R]) -> tuple[NDArray, R]:
        """Combine poses of frame 1 wrt 0 and 2 wrt 1 into 2 wrt 0."""
        tvec02 = pose01[0] + pose01[1].apply(pose12[0])
        rot02 = pose01[1] * pose12[1]
        return tvec02, rot02

    def _get_elapsed(self) -> float:
        """Get the elapsed time in seconds since the last trajectory."""
        s, ns = (self.get_clock().now() - self._traj_stamp).seconds_nanoseconds()
        return s + ns * 1e-9

    def _solve_for_trajectory(self, pose_mk_cam: NDArray, elaps: float) -> tuple[NDArray, NDArray]:
        """Use OCP solver to solve for reference trajectory.

        Args:
            pose_mk_cam: Latest camera pose wrt reference marker.
            elaps: Elapsed time in seconds since last trajectory call.

        Returns:
            tuple containing camera pose and twist sequences respectively.
        """
        # combine solver initial state and reference pose
        x0 = np.concatenate([pose_mk_cam, self._twist])
        ref = np.concatenate(
            [self._pose_mk_camd[0], self._pose_mk_camd[1].as_quat(scalar_first=True)]
        )
        # reset solver if new reference was set
        if self._pose_tgt_tcpd is not None:
            self._ocp_solver.reset(x0)
        else:
            self._ocp_solver.warmup(n_shift=int(elaps / self._time_step))
        # solve for trajectory
        pose, twist = self._ocp_solver.solve(x0, ref)
        return pose, twist

    def _reset(self) -> None:
        """Reset all attributes that are initialized from callbacks."""
        self._pose_tgt_tcpd = None
        self._pose_mk_camd = None
        self._twist = np.zeros(6)
        self._ocp_solver = None


def main(args=None):
    rclpy.init(args=args)
    oc_planner = OcPlanner()
    rclpy.spin(oc_planner)
    oc_planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
