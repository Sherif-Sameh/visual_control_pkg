#!/usr/bin/env python3

"""
ROS node for performing hyperparameter sweeps using WandB.
"""

from typing import Any

import numpy as np
import rclpy
import toml
import wandb
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Header

from visual_control_pkg.loggers import ROSWrapperLogger
from visual_control_pkg.loggers.wandb import WandBLogger
from visual_control_pkg.metrics import (
    AccumulatorMetric,
    ComposeMetric,
    DeltaMetric,
    FunctionalMetric,
)
from visual_control_pkg.objectives import NormObjective, RotNormObjective, TfNormObjective
from visual_control_pkg.utils.common import process_wandb_config
from visual_control_pkg.utils.ros import python_to_param_value


class ROSSweeper(Node):
    """ROS node for performing hyperparameter sweeps using WandB."""

    def __init__(self):
        super().__init__("ros_sweeper")

        # Declare ROS parameters
        self.declare_parameter("timer_period", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("n_runs", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("n_runs_per_set", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("target_nodes", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("obj.flags", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("obj.weights", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("sweep.id", rclpy.Parameter.Type.STRING)
        self.declare_parameter("sweep.config", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.n_log", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("wandb.n_flush", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("wandb.config.entity", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.project", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.group", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.dir", rclpy.Parameter.Type.STRING)
        self.declare_parameter("wandb.config.params", rclpy.Parameter.Type.STRING_ARRAY)

        # Initialize non-ROS class attributes
        timer_period = self.get_parameter("timer_period").value
        self._n_runs_left = self.get_parameter("n_runs").value
        self._n_runs_left = float("inf") if self._n_runs_left <= 0 else self._n_runs_left
        self._n_runs_per_set = self.get_parameter("n_runs_per_set").value
        self._n_runs_per_set = 1 if self._n_runs_per_set <= 0 else self._n_runs_per_set
        self._n_runs_per_set_left = self._n_runs_per_set
        self._target_nodes: list[str] = self.get_parameter("target_nodes").value
        self._obj_flags = self.get_parameter("obj.flags").value
        self._obj_weights = np.array(self.get_parameter("obj.weights").value)
        self._sweep_params = {k: v.value for k, v in self.get_parameters_by_prefix("sweep").items()}
        self._sweep_params["config"] = toml.load(self._sweep_params["config"])["sweep"]
        self._obj = AccumulatorMetric(name="objective", argname="objective", red="mean")
        self._metrics = (self._init_js_metric(), self._init_pe_metric())
        self._metrics_t = [0.0, 0.0]
        self._metrics_nan = [True, True]
        self._wandb: WandBLogger | None = None
        self._wandb_kwargs: dict[str, Any] = {}
        self._start_time = None
        assert len(self._obj_flags) == len(self._obj_weights)

        # Initialize ROS attributes
        self._sub_js = self.create_subscription(JointState, "/joint_states", self.callback_js, 50)
        self._sub_pe = self.create_subscription(PoseStamped, "/pose_error", self.callback_pe, 50)
        self._sub_rst = self.create_subscription(
            Empty, "/ros_sweeper/restart", self.callback_rst, 10
        )
        self._get_cli = [
            self.create_client(GetParameters, f"/{tn}/get_parameters") for tn in self._target_nodes
        ]
        self._set_cli = [
            self.create_client(SetParameters, f"/{tn}/set_parameters") for tn in self._target_nodes
        ]
        for tn, get_c in zip(self._target_nodes, self._get_cli):
            while not get_c.wait_for_service(timeout_sec=0.5):
                self.get_logger().info(
                    f"Get parameters service for {tn} not available, waiting again..."
                )
        self._timer = self.create_timer(timer_period, self.callback_timer)

    @property
    def run_set_done(self) -> bool:
        """Returns flag indentifying whether the current set of runs is done or not."""
        return self._n_runs_per_set_left <= 0

    @property
    def sweep_id(self) -> str:
        """Returns the current sweep ID.

        Will return the node's current sweep ID if it is preset. If not, it will initialize a new
        sweep and return its ID.
        """
        if self._sweep_params["id"] == "":
            self._sweep_params["id"] = wandb.sweep(
                sweep=self._sweep_params["config"],
                project=self.get_parameter("wandb.config.project").value,
            )
        return self._sweep_params["id"]

    @property
    def agent_kwargs(self) -> dict[str, Any]:
        """Returns the kwargs for the wandb agent.

        Those include the count (number of sweeps), entity and project.
        """
        kwargs = {
            "count": self.get_parameter("n_runs").value,
            "entity": self.get_parameter("wandb.config.entity").value,
            "project": self.get_parameter("wandb.config.project").value,
        }
        return kwargs

    def pre_run_set(self) -> None:
        """Setup routine to prepare for a new set of runs in the sweep."""
        if self._n_runs_left <= 0:
            return

        # Initialize/re-intiialize WandB logger
        if self._wandb is None:
            self._wandb_kwargs = self._init_wandb_kwargs()
        self._wandb = ROSWrapperLogger(n_hold=2, logger=WandBLogger(**self._wandb_kwargs))

        # Send hyperparameters to target nodes
        config = process_wandb_config(wandb.config.as_dict())
        self._set_hyperparameters(config)
        self._start_time = self.get_clock().now().nanoseconds * 1e-9

    def post_run_set(self) -> None:
        """Termination routine to close an active set of runs in the sweep cleanly."""
        # Log objective value for run set
        step = max(self._metrics_t[0], self._metrics_t[1])
        obj_value = self._obj.compute()
        for _ in range(self._wandb_kwargs["n_log"]):
            self._wandb.log(step, {self._obj.name: obj_value})
            self._wandb.log(step, {})  # counteract n_hold=2

        # Reset internal states to prepare for new run set
        self._obj.reset()
        self._wandb.close()
        self._n_runs_per_set_left = self._n_runs_per_set

    def callback_timer(self) -> None:
        if self._wandb is None or any(self._metrics_nan):
            return
        terms = []
        for i, (step, metric) in enumerate(zip(self._metrics_t, self._metrics)):
            self._wandb.log(step, metrics_dict := metric.compute())
            terms.extend(metrics_dict.values())
            metric.reset()
            self._metrics_nan[i] = True
        obj_value = np.sum(self._obj_weights * np.concatenate(terms, axis=0), keepdims=True)
        self._obj.update(objective=obj_value)

    def callback_js(self, msg: JointState) -> None:
        self._metrics[0].update(velocity=np.array(msg.velocity))
        self._metrics_t[0] = self._get_timestep(msg.header)
        self._metrics_nan[0] = False

    def callback_pe(self, msg: PoseStamped) -> None:
        pos, rot = msg.pose.position, msg.pose.orientation
        self._metrics[1].update(
            position=np.array([pos.x, pos.y, pos.z]),
            rotation=np.array([rot.x, rot.y, rot.z, rot.w]),
            pose=np.array([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w]),
        )
        self._metrics_t[1] = self._get_timestep(msg.header)
        self._metrics_nan[1] = False

    def callback_rst(self, msg: Empty) -> None:
        self._n_runs_per_set_left -= 1
        if self._n_runs_per_set_left == 0:
            self._n_runs_left -= 1
        if self._n_runs_left > 0:
            for i, metric in enumerate(self._metrics):
                metric.reset()
                self._metrics_nan[i] = True

    def _init_js_metric(self) -> ComposeMetric:
        """Initialize and return composed tracked JS metrics according to objective configuration."""
        metrics = []
        if "jnt_vel" in self._obj_flags:
            metrics.append(
                FunctionalMetric(
                    name="JS/VelL2Norm",
                    metric=AccumulatorMetric(name="", argname="velocity", red="mean"),
                    func=NormObjective(name="", ord=2),
                )
            )
        if "jnt_delta_vel" in self._obj_flags:
            metrics.append(
                DeltaMetric(
                    name="JS/DeltaVelL2Norm",
                    argname="velocity",
                    metric=FunctionalMetric(
                        name="",
                        metric=AccumulatorMetric(name="", argname="", red="mean"),
                        func=NormObjective(name="", ord=2),
                    ),
                    default=np.zeros(1),
                )
            )
        return ComposeMetric(metrics=metrics)

    def _init_pe_metric(self) -> ComposeMetric:
        """Initialize and return composed tracked PE metrics according to objective configuration."""
        metrics = []
        if "pos" in self._obj_flags:
            metrics.append(
                FunctionalMetric(
                    name="PE/PosL2Norm",
                    metric=AccumulatorMetric(name="", argname="position", red="mean"),
                    func=NormObjective(name="", ord=2),
                )
            )
        if "rot" in self._obj_flags:
            metrics.append(
                FunctionalMetric(
                    name="PE/RotL2Norm",
                    metric=AccumulatorMetric(name="", argname="rotation", red="mean"),
                    func=RotNormObjective(name="", ord=2, repr="quat"),
                )
            )
        if "pose" in self._obj_flags:
            metrics.append(
                FunctionalMetric(
                    name="PE/PoseL2Norm",
                    metric=AccumulatorMetric(name="", argname="pose", red="mean"),
                    func=TfNormObjective(name="", ord=2, repr="quat"),
                )
            )
        return ComposeMetric(metrics=metrics)

    def _init_wandb_kwargs(self) -> dict[str, Any]:
        """Initialize and return the WandBLogger's kwargs according to its declared parameters.

        After reading the logger's parameters, the function attempts to get the values of the
        parameter's given to the logger as part of its config.
        """
        params = {k: v.value for k, v in self.get_parameters_by_prefix("wandb").items()}
        param_names: list[str] = params["config.params"]
        param_dict = {}
        for tn, get_c in zip(self._target_nodes, self._get_cli):
            request = GetParameters.Request()
            request.names = [n.split("-")[-1] for n in param_names if n.startswith(tn)]
            future = get_c.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response: GetParameters.Response = future.result()
            param_dict.update(
                {k: parameter_value_to_python(v) for k, v in zip(request.names, response.values)}
            )
        config = WandBLogger.WandBConfig(
            entity=None,
            project=None,
            group=params["config.group"],
            dir=params["config.dir"],
            config=param_dict,
        )
        wandb_kwargs = {k: v for k, v in params.items() if not k.startswith("config")}
        wandb_kwargs["config"] = config
        return wandb_kwargs

    def _set_hyperparameters(self, config: dict[str, Any]) -> None:
        """Set given hyperparameters to target nodes through their `set_parameters()` services.

        Args:
            config: Dictionary of hyperparameters to send to target nodes. Each parameter's key
                must include the nodes name prior to the parameter's name (e.g. `/controller/gain`).
        """
        for tn, set_c in zip(self._target_nodes, self._set_cli):
            request = SetParameters.Request()
            request.parameters = [
                Parameter(name=k.split("-")[-1], value=python_to_param_value(v))
                for k, v in config.items()
                if k.startswith(tn)
            ]
            if len(request.parameters) == 0:
                continue
            future = set_c.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response: SetParameters.Response = future.result()
            for i, r in enumerate(response.results):
                if not r.successful:
                    self.get_logger().warn(
                        f"Set {request.parameters[i].name} failed due to {r.reason}"
                    )

    def _get_timestep(self, header: Header) -> float:
        """Get the current timestep for the provided header since the node's initialization."""
        h_time = header.stamp.sec + header.stamp.nanosec * 1e-9
        return h_time - self._start_time


def main(args=None):
    rclpy.init(args=args)
    ros_sweeper = ROSSweeper()
    sweep_id = ros_sweeper.sweep_id
    agent_kwargs = ros_sweeper.agent_kwargs

    def sweep_fn() -> None:
        ros_sweeper.pre_run_set()
        while rclpy.ok() and not ros_sweeper.run_set_done:
            rclpy.spin_once(ros_sweeper)
        ros_sweeper.post_run_set()

    wandb.agent(sweep_id, sweep_fn, **agent_kwargs)
    ros_sweeper.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
