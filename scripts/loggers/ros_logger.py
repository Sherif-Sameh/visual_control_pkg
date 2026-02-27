#!/usr/bin/env python3

"""
ROS node for logging metrics during runtime from active topics.
"""

from functools import partial

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Header
from trajectory_msgs.msg import JointTrajectory

from visual_control_pkg.loggers import ComposeLogger, ConsoleLogger, ROSWrapperLogger
from visual_control_pkg.loggers.csv import CSVLogger
from visual_control_pkg.loggers.wandb import WandBLogger
from visual_control_pkg.metrics import AccumulatorMetric, ComposeMetric, UnitMetric


class ROSLogger(Node):
    """ROS node for logging metrics during runtime from active topics.

    The logger supports any and all of the three possible outputs of console, CSV file and WandB.
    The configuration of topics is fixed as their types have to be known for the callbacks.
    However, the configuration of the loggers themselves is set through the node's parameters.

    The following topics are supported for logging if they're available:
        - Joint state (sensor_msgs/JointState): Joint position, velocity and effort measurements.

        - Joint trajectory (trajectory_msgs/JointTrajectory): Joint position, velocity,
        acceleration and effort actuations.

        - Pose error (geometry_msgs/PoseStamped): Pose tracking error for the manipulator's
        end-effector.

        - Setpoint errors (geometry_msgs/PoseArray): Control setpoint pose tracking errors as
        measured by the active controller (i.e. control driving signal).

    Also, the node has an optional restart topic (std_msgs/Empty) for triggering a restart across
    all its active loggers.
    """

    def __init__(self):
        super().__init__("ros_logger")

        # Declare ROS parameters
        self.declare_parameter("n_runs", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("smooth", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("timer_period", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("param_servers", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("log.console", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("log.csv", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("log.wandb", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("console.n_log", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("console.n_flush", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("console.filter", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("console.config.precision", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("console.config.separator", rclpy.Parameter.Type.STRING)
        self.declare_parameter("console.config.sign", rclpy.Parameter.Type.STRING)
        self.declare_parameter("csv.n_log", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("csv.n_flush", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("csv.filter", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("csv.dir", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.n_log", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("wandb.n_flush", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("wandb.filter", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("wandb.config.entity", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.project", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.group", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.dir", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.params", rclpy.Parameter.Type.STRING_ARRAY)

        # Initialize non-ROS class attributes
        timer_period = self.get_parameter("timer_period").value
        n_runs = self.get_parameter("n_runs").value
        self._n_runs_left = float("inf") if n_runs <= 0 else n_runs
        self._param_servers: list[str] = self.get_parameter("param_servers").value
        self._log_flags = {k: v.value for k, v in self.get_parameters_by_prefix("log").items()}
        if not any(f for f in self._log_flags.values()):
            self.get_logger().info("None of the logging outputs are enabled.")
            self.get_logger().info("Shutting down.")
            rclpy.shutdown()
        self._metrics: dict[str, list[float | ComposeMetric]] = dict()
        self._loggers: ComposeLogger | None = None
        self._start_time = self.get_clock().now().nanoseconds * 1e-9

        # Initialize ROS attributes
        self._sub_js = self.create_subscription(JointState, "/joint_states", self.callback_js, 50)
        self._sub_jt = self.create_subscription(
            JointTrajectory, "/joint_trajectory", self.callback_jt, 50
        )
        self._sub_pe = self.create_subscription(PoseStamped, "/pose_error", self.callback_pe, 50)
        self._sub_se = self.create_subscription(PoseArray, "/setpoint_error", self.callback_se, 50)
        self._sub_rst = self.create_subscription(
            Empty, "/ros_logger/restart", self.callback_rst, 10
        )
        self._timer = self.create_timer(timer_period, self.callback_timer)
        if self._log_flags["wandb"]:
            self._cli = [self.create_client(GetParameters, ps) for ps in self._param_servers]
            for ps, c in zip(self._param_servers, self._cli):
                while not c.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(f"{ps} service not available, waiting again...")

    @property
    def shutdown(self) -> bool:
        """Returns flag indentifying whether the node should shutdown or not."""
        return self._n_runs_left <= 0

    def post_init(self) -> None:
        """Initialize all loggers."""
        loggers = []
        if self._log_flags["console"]:
            loggers.append(self._init_console_logger())
            self.get_logger().info("Initialized console logger.")
        if self._log_flags["csv"]:
            loggers.append(self._init_csv_logger())
            self.get_logger().info("Initialized CSV logger.")
        if self._log_flags["wandb"]:
            loggers.append(self._init_wandb_logger())
            self.get_logger().info("Initialized WandB logger.")
        self._loggers = ComposeLogger(loggers=loggers)

    def close(self) -> None:
        """Close all active loggers cleanly."""
        self._loggers.close()

    def callback_timer(self) -> None:
        for step, metric in self._metrics.values():
            self._loggers.log(step, metric.compute())
            metric.reset()

    def callback_js(self, msg: JointState) -> None:
        if "js" not in self._metrics:
            self._metrics["js"] = [0, self._init_js_metric(msg)]

        self._metrics["js"][0] = self._get_timestep(msg.header)
        self._metrics["js"][1].update(
            position=np.array(msg.position),
            velocity=np.array(msg.velocity),
            effort=np.array(msg.effort),
        )

    def callback_jt(self, msg: JointTrajectory) -> None:
        assert len(msg.points) > 0
        if "jt" not in self._metrics:
            self._metrics["jt"] = [0, self._init_jt_metric(msg)]

        self._metrics["jt"][0] = self._get_timestep(msg.header)
        self._metrics["jt"][1].update(
            position=np.array(msg.points[0].positions),
            velocity=np.array(msg.points[0].velocities),
            acceleration=np.array(msg.points[0].accelerations),
            effort=np.array(msg.points[0].effort),
        )

    def callback_pe(self, msg: PoseStamped) -> None:
        if "pe" not in self._metrics:
            self._metrics["pe"] = [0, self._init_pe_metric(msg)]

        pos, rot = msg.pose.position, msg.pose.orientation
        self._metrics["pe"][0] = self._get_timestep(msg.header)
        self._metrics["pe"][1].update(
            position=np.array([pos.x, pos.y, pos.z]),
            rotvec=R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_rotvec(),
        )

    def callback_se(self, msg: PoseArray) -> None:
        if "se" not in self._metrics:
            self._metrics["se"] = [0, self._init_se_metric(msg)]

        kwargs = {}
        for i, pose in enumerate(msg.poses):
            pos, rot = pose.position, pose.orientation
            kwargs[f"position_{i}"] = np.array([pos.x, pos.y, pos.z])
            kwargs[f"rotvec_{i}"] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_rotvec()
        self._metrics["se"][0] = self._get_timestep(msg.header)
        self._metrics["se"][1].update(**kwargs)

    def callback_rst(self, msg: Empty) -> None:
        self._n_runs_left -= 1
        if not self.shutdown:
            self._loggers.restart()
            for _, metric in self._metrics.values():
                metric.reset()
            self._start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info("Restarted ROS logger.")

    def _get_timestep(self, header: Header) -> float:
        """Get the current timestep for the provided header since the node's initialization."""
        h_time = header.stamp.sec + header.stamp.nanosec * 1e-9
        return h_time - self._start_time

    def _init_console_logger(self) -> ROSWrapperLogger:
        """Initialize and return a ConsoleLogger according to its declared parameters."""
        params = {k: v.value for k, v in self.get_parameters_by_prefix("console").items()}
        return ROSWrapperLogger(
            n_hold=4,
            logger=ConsoleLogger(
                n_log=params["n_log"],
                n_flush=params["n_flush"],
                filter=params["filter"] if params["filter"][0] != "" else None,
                config=ConsoleLogger.ArrayPrintOptions(
                    precision=params["config.precision"],
                    separator=params["config.separator"],
                    sign=params["config.sign"],
                ),
            ),
        )

    def _init_csv_logger(self) -> ROSWrapperLogger:
        """Initialize and return a CSVLogger according to its declared parameters."""
        params = {k: v.value for k, v in self.get_parameters_by_prefix("csv").items()}
        return ROSWrapperLogger(
            n_hold=4,
            logger=CSVLogger(
                n_log=params["n_log"],
                n_flush=params["n_flush"],
                filter=params["filter"] if params["filter"][0] != "" else None,
                dir=params["dir"],
            ),
        )

    def _init_wandb_logger(self) -> ROSWrapperLogger:
        """Initialize and return a WandB according to its declared parameters.

        After reading the logger's parameters, the function attempts to get the values of the
        parameter's given to the logger as part of its config.

        **Warning**: This function must be called after the node has been initialized since it
        needs to request parameters from the GetParameter server.
        """
        params = {k: v.value for k, v in self.get_parameters_by_prefix("wandb").items()}
        param_names: list[str] = params["config.params"]
        param_dict: dict[str, str] = {}
        for ps, c in zip(self._param_servers, self._cli):
            request = GetParameters.Request()
            request.names = [
                n.split("/")[-1] for n in param_names if n.startswith(ps.split("/")[0])
            ]
            future = c.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response: GetParameters.Response = future.result()
            param_dict.update(
                {k: parameter_value_to_python(v) for k, v in zip(request.names, response.values)}
            )
        return ROSWrapperLogger(
            n_hold=4,
            logger=WandBLogger(
                n_log=params["n_log"],
                n_flush=params["n_flush"],
                filter=params["filter"] if params["filter"][0] != "" else None,
                config=WandBLogger.WandBConfig(
                    entity=params["config.entity"],
                    project=params["config.project"],
                    group=params["config.group"],
                    dir=params["config.dir"],
                    config=param_dict,
                ),
            ),
        )

    def _init_js_metric(self, msg: JointState) -> ComposeMetric:
        """Initialize the JointState metric according the sample message."""
        smooth = self.get_parameter("smooth").value
        cls = partial(AccumulatorMetric, red="mean") if smooth else UnitMetric
        metrics = []
        if len(msg.position) > 0:
            metrics.append(cls(name="JS (Position)", argname="position"))
        if len(msg.velocity) > 0:
            metrics.append(cls(name="JS (Velocity)", argname="velocity"))
        if len(msg.effort) > 0:
            metrics.append(cls(name="JS (Effort)", argname="effort"))
        return ComposeMetric(metrics=metrics)

    def _init_jt_metric(self, msg: JointTrajectory) -> ComposeMetric:
        """Initialize the JointTrajectory metric according the sample message."""
        smooth = self.get_parameter("smooth").value
        cls = partial(AccumulatorMetric, red="mean") if smooth else UnitMetric
        metrics = []
        if len(msg.points[0].positions) > 0:
            metrics.append(cls(name="JT (Position)", argname="position"))
        if len(msg.points[0].velocities) > 0:
            metrics.append(cls(name="JT (Velocity)", argname="velocity"))
        if len(msg.points[0].accelerations) > 0:
            metrics.append(cls(name="JT (Acceleration)", argname="acceleration"))
        if len(msg.points[0].effort) > 0:
            metrics.append(cls(name="JT (Effort)", argname="effort"))
        return ComposeMetric(metrics=metrics)

    def _init_pe_metric(self, msg: PoseStamped) -> ComposeMetric:
        smooth = self.get_parameter("smooth").value
        cls = partial(AccumulatorMetric, red="mean") if smooth else UnitMetric
        metrics = [
            cls(name="PE (Position)", argname="position"),
            cls(name="PE (RotVec)", argname="rotvec"),
        ]
        return ComposeMetric(metrics=metrics)

    def _init_se_metric(self, msg: PoseArray) -> ComposeMetric:
        smooth = self.get_parameter("smooth").value
        cls = partial(AccumulatorMetric, red="mean") if smooth else UnitMetric
        metrics = []
        for i in range(len(msg.poses)):
            metrics.append(cls(name=f"SE_{i} (Position)", argname=f"position_{i}"))
            metrics.append(cls(name=f"SE_{i} (RotVec)", argname=f"rotvec_{i}"))
        return ComposeMetric(metrics=metrics)


def main(args=None):
    rclpy.init(args=args)
    ros_logger = ROSLogger()
    ros_logger.post_init()
    while rclpy.ok() and not ros_logger.shutdown:
        rclpy.spin_once(ros_logger)
    if ros_logger.shutdown:
        ros_logger.get_logger().info("Completed planned runs.")
        ros_logger.get_logger().info("Shutting down.")
    ros_logger.close()
    ros_logger.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
