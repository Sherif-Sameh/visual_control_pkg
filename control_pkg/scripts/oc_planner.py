#!/usr/bin/env python3

"""
ROS node for optimal control-based trajectory planning using Acados
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Transform, Twist, TwistStamped
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from numpy.typing import NDArray
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Header
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

import vc_core.utils.geometry.pose as pose_utils
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
        self.declare_parameter("tag.tag_id", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("pose.mk_tgt", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("twist.tol", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("model.alpha", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("model.fp", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("cost.Q_x_diag", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("cost.R_u_diag", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("cost.Q_z_diag", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.lbx", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.ubx", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.idxbx", rclpy.Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter("constraint.lbu", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.ubu", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("constraint.idxbu", rclpy.Parameter.Type.INTEGER_ARRAY)
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
        self._tag_id = self.get_parameter("tag.tag_id").value
        self._pose_mk_tgt = (
            np.array(pose_mk_tgt[:3]),
            R.from_quat(pose_mk_tgt[3:], scalar_first=True),
        )
        self._twist_tol = self.get_parameter("twist.tol").value
        self._con_lbx = np.array(self.get_parameter("constraint.lbx").value)
        self._con_ubx = np.array(self.get_parameter("constraint.ubx").value)
        self._time_step = self.get_parameter("solver.time_step").value
        self._traj_stamp = self.get_clock().now()
        self._ocp_solver = self._init_ocp_solver()
        self._pose_tcp_cam = None
        self._pose_tgt_tcpd = None
        self._pose_mk_camd = None
        self._twist = np.zeros(6)
        self._twist_ref = np.zeros(6)

        # Initialize ROS attributes
        self._pub_traj = self.create_publisher(MultiDOFJointTrajectory, "/oc_planner/trajectory", 0)
        self._pub_exec_time = self.create_publisher(Float64, "/oc_planner/execution_time", 0)
        self._sub_sref = self.create_subscription(
            MultiDOFJointTrajectory, "/state_reference", self.callback_sref, 1
        )
        self._sub_cam_twist = self.create_subscription(
            TwistStamped, "/camera_twist", self.callback_cam_twist, 0
        )
        self._sub_dtn = self.create_subscription(
            AprilTagDetectionArray, "/detections", self.callback_dtn, 0
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def publish_traj(self, header: Header, pose: NDArray, twist: NDArray) -> None:
        """Publish the planned trajectory."""
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = header.stamp

        traj_len = min(int(1.5 * self._n_update), pose.shape[0])
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

    def callback_sref(self, msg: MultiDOFJointTrajectory) -> None:
        t: Transform = msg.points[0].transforms[0]
        tw: Twist = msg.points[0].velocities[0]
        self._pose_tgt_tcpd = pose_utils.from_transform_gm(t)
        self._twist_ref[:3] = tw.linear.x, tw.linear.y, tw.linear.z
        self._twist_ref[3:] = tw.angular.x, tw.angular.y, tw.angular.z
        # if np.allclose(self._twist_ref, np.zeros(6), atol=1e-3):
        #     # reset twist constraints
        #     self._ocp_solver.set_constraints(["lbx", "ubx"], [self._con_lbx, self._con_ubx])

    def callback_cam_twist(self, msg: TwistStamped) -> None:
        self._twist[:3] = msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
        self._twist[3:] = msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z

    def callback_dtn(self, msg: AprilTagDetectionArray) -> None:
        # resolve camera wrt TCP transform
        if self._pose_tcp_cam is None:
            self._pose_tcp_cam = self._lookup_transform()
        no_ref = self._pose_tgt_tcpd is None and self._pose_mk_camd is None
        if no_ref or self._pose_tcp_cam is None:
            return
        elapsed = self._get_elapsed()
        if self._pose_tgt_tcpd is None and elapsed < self._n_update * self._time_step:
            return
        # compute camera pose wrt marker
        dtn = [dtn for dtn in msg.detections if dtn.id == self._tag_id]
        if not dtn:
            return
        pose = pose_utils.pose_inv(pose_utils.from_pose_gm(dtn[0].pose.pose.pose))
        pose_mk_cam = np.concatenate([pose[0], pose[1].as_quat(scalar_first=True)])
        # update reference if a new reference is available
        if self._pose_tgt_tcpd is not None:
            pose_tgt_camd = pose_utils.pose_mult(self._pose_tgt_tcpd, self._pose_tcp_cam)
            self._pose_mk_camd = pose_utils.pose_mult(self._pose_mk_tgt, pose_tgt_camd)
        # get and publish new trajectory
        pose, twist = self._solve_for_trajectory(pose_mk_cam, elapsed)
        self.publish_traj(msg.header, pose, twist)
        self.publish_exec_time(self._ocp_solver.get_stats("time_tot"))
        self._pose_tgt_tcpd = None
        self._twist_ref = np.zeros(6)
        self._traj_stamp = self.get_clock().now()

    def _init_ocp_solver(self) -> VsOcpSolver:
        """Initialize the VS OCP solver."""
        model_params = {k: v.value for k, v in self.get_parameters_by_prefix("model").items()}
        cost_params = {k: v.value for k, v in self.get_parameters_by_prefix("cost").items()}
        constraint_params = {
            k: np.array(v.value) for k, v in self.get_parameters_by_prefix("constraint").items()
        }
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
        return pose_utils.from_transform_gm(t.transform)

    def _get_elapsed(self) -> float:
        """Get the elapsed time in seconds since the last trajectory."""
        ns = (self.get_clock().now() - self._traj_stamp).nanoseconds
        return ns * 1e-9

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
        # tighten twist constraints if twist reference is given
        # if not np.allclose(self._twist_ref, np.zeros(6), atol=1e-3):
        #     twist_ref = np.max(
        #         np.abs(
        #             pose_utils.pose_adj(pose_utils.pose_inv(self._pose_tcp_cam)) @ self._twist_ref
        #         )
        #     )
        #     con_lbx = np.maximum(self._con_lbx, -twist_ref)
        #     con_ubx = np.minimum(self._con_ubx, twist_ref)
        #     self._ocp_solver.set_constraints(["lbx", "ubx"], [con_lbx, con_ubx])
        # solve for trajectory
        pose, twist = self._ocp_solver.solve(x0, ref)
        return pose, twist


def main(args=None):
    rclpy.init(args=args)
    oc_planner = OcPlanner()
    rclpy.spin(oc_planner)
    oc_planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
