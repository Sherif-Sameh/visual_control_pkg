from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver

from vc_core.ocp.model import export_vs_ode_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class VsOcpSolverCfg:
    """Configuration for the task-space visual servoing Acados OCP solver."""

    @dataclass
    class CostCfg:
        """Quadratic cost function configuration."""

        Q_x: np.ndarray
        """Cost matrix for the model state errors (`p`, `q`, `u`). Shape is (12, 12)."""

        R_u: np.ndarray
        """Cost matrix for the model inputs (`u_dot`). Shape is (6, 6)."""

        Q_z: np.ndarray
        """Cost matrix for the model feature errors (`s_dot`). Shape is (2, 2)."""

        Q_x_e: np.ndarray
        """Cost matrix for the model terminal state errors (`p`, `q`, `u`). Shape is (12, 12)."""

    cost_cfg: CostCfg
    """Quadratic cost function configuration."""

    alpha: float = 1.0
    """Coefficient of quat norm restorative term in equation for `qdot`. Default value is 1.0."""

    fp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Visual feature coordinates in the marker's refernce frame. Default value is zeros(3)."""

    @dataclass
    class ConstraintCfg:
        """Model state and input constraint configuration."""

        lbx: np.ndarray = field(default_factory=lambda: np.array([]))
        """Model state lower bound. Default is empty array (i.e. no bound)."""

        ubx: np.ndarray = field(default_factory=lambda: np.array([]))
        """Model state upper bound. Default is empty array (i.e. no bound)."""

        idxbx: np.ndarray = field(default_factory=lambda: np.array([]))
        """Indices of model state elements bound by entries in `lbx` and `ubx`. Default is empty array
        (i.e. no bound).
        """

        lbu: np.ndarray = field(default_factory=lambda: np.array([]))
        """Model input lower bound. Default is empty array (i.e. no bound)."""

        ubu: np.ndarray = field(default_factory=lambda: np.array([]))
        """Model input upper bound. Default is empty array (i.e. no bound)."""

        idxbu: np.ndarray = field(default_factory=lambda: np.array([]))
        """Indices of model input elements bound by entries in `lbu` and `ubu`. Default is empty array
        (i.e. no bound).
        """

        lh: np.ndarray = field(default_factory=lambda: np.array([]))
        """Model nonlinear visibility constraint lower bound. Default is empty array (i.e. no bound)."""

        uh: np.ndarray = field(default_factory=lambda: np.array([]))
        """Model nonlinear visibility constraint upper bound. Default is empty array (i.e. no bound)."""

    constraint_cfg: ConstraintCfg = ConstraintCfg()
    """Model state and input constraint configuration. Default values derived from `ConstraintCfg`."""

    @dataclass
    class SolverCfg:
        n_horizon: int = 40
        """Number of time-steps for forward prediction horizon. Default value is 40."""

        time_step: float = 0.1
        """Time step for prediction horizon in seconds. Default value is 0.1."""

        nlp_solver_max_iter: int = 100
        """NLP solver maximum number of iterations. Default value is 100."""

        nlp_tol: float = 1e-5
        """NLP solver tolerance. Default value is 1e-5."""

        integrator_type: str = "ERK"
        """Integrator type. Default value is ERK (Explicit Runge Kutta)."""

        levenberg_marquardt: float = 0.0
        """Factor for LM regularization. Default value is 0.0."""

        qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
        """QP solver to be used in the NLP solver. Default value is PARTIAL_CONDENSING_HPIPM."""

        qp_solver_iter_max: int = 50
        """QP solver: maximum number of iterations. Default value is 50."""

        qp_tol: float = 1e-5
        """QP solver tolerance. Default value is 1e-5."""

    solver_cfg: SolverCfg = SolverCfg()
    """Solver configuration. Default values derived from `SolverCfg`."""


class VsOcpSolver:
    """Task-space visual servoing Acados OCP solver.

    For more details on the model used, refer to `vc_core.ocp.model.export_vs_ode_model`.

    Args:
        cfg: Configuration for the Acados OCP solver.
    """

    NX = 13  # Size of state vector
    NU = 6  # Size of input vector
    NZ = 4  # Size of feature vector

    def __init__(self, cfg: VsOcpSolverCfg):
        self._check_dims(cfg)
        # setup ocp and model
        self._ocp = AcadosOcp()
        self._ocp.model = export_vs_ode_model(cfg.fp, alpha=cfg.alpha)
        self._ocp.parameter_values = np.zeros(7)

        # setup cost
        self._ocp.cost.cost_type = "NONLINEAR_LS"
        self._ocp.cost.cost_type_e = "NONLINEAR_LS"
        self._ocp.cost.W = scipy.linalg.block_diag(
            cfg.cost_cfg.Q_x, cfg.cost_cfg.R_u, cfg.cost_cfg.Q_z
        )
        self._ocp.cost.W_e = cfg.cost_cfg.Q_x_e
        self._ocp.cost.yref = np.zeros(self.NX - 1 + self.NU + self.NZ // 2)
        self._ocp.cost.yref_e = np.zeros(self.NX - 1)

        # setup constraints
        self._ocp.constraints.x0 = np.zeros(self.NX)
        self._ocp.constraints.lbx = cfg.constraint_cfg.lbx
        self._ocp.constraints.ubx = cfg.constraint_cfg.ubx
        self._ocp.constraints.idxbx = cfg.constraint_cfg.idxbx
        self._ocp.constraints.lbu = cfg.constraint_cfg.lbu
        self._ocp.constraints.ubu = cfg.constraint_cfg.ubu
        self._ocp.constraints.idxbu = cfg.constraint_cfg.idxbu
        self._ocp.constraints.lh = cfg.constraint_cfg.lh
        self._ocp.constraints.uh = cfg.constraint_cfg.uh

        # setup solver
        self._ocp.solver_options.N_horizon = cfg.solver_cfg.n_horizon
        self._ocp.solver_options.tf = cfg.solver_cfg.n_horizon * cfg.solver_cfg.time_step
        self._ocp.solver_options.nlp_solver_max_iter = cfg.solver_cfg.nlp_solver_max_iter
        self._ocp.solver_options.tol = cfg.solver_cfg.nlp_tol
        self._ocp.solver_options.integrator_type = cfg.solver_cfg.integrator_type
        self._ocp.solver_options.levenberg_marquardt = cfg.solver_cfg.levenberg_marquardt
        self._ocp.solver_options.qp_solver = cfg.solver_cfg.qp_solver
        self._ocp.solver_options.qp_solver_iter_max = cfg.solver_cfg.qp_solver_iter_max
        self._ocp.solver_options.qp_tol = cfg.solver_cfg.qp_tol
        self._ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

        json_file = "acados_ocp_" + self._ocp.model.name + ".json"
        self._ocp.code_gen_opts.code_export_directory = "c_generated_code_ocp"
        self._ocp.code_gen_opts.json_file = json_file
        self._ocp_solver = AcadosOcpSolver(self._ocp, json_file=json_file)

    @property
    def solver(self) -> AcadosOcpSolver:
        """Get underlying OCP solver."""
        return self._ocp_solver

    def reset(self, x0: NDArray) -> None:
        """Reset the solver's state sequence to `x0` and input sequence to zero.

        Args:
            x0: Initial state. Shape is (`NX`,).
        """
        n_horizon = self._ocp.solver_options.N_horizon
        self._ocp_solver.set_flat("x", np.tile(x0, n_horizon + 1))
        self._ocp_solver.set_flat("u", np.zeros(self.NU * n_horizon))

    def set_constraints(self, fields: list[str], values: list[NDArray]) -> None:
        """Set the values of the solver's constraints for all intermediate stages.

        Args:
            fields: List of constraint attributes to set (e.g. lbx, ubx, etc.).
            values: Corresponding values of `fields`.
        """
        n_horizon = self._ocp.solver_options.N_horizon
        for i in range(1, n_horizon):
            for f, v in zip(fields, values):
                self._ocp_solver.constraints_set(i, f, v)

    def solve(
        self, x0: NDArray, ref: NDArray, print_stats_on_failure: bool = False
    ) -> tuple[NDArray, NDArray]:
        """Solve NLP for control input sequence.

        Args:
            x0: Initial state. Shape is (`NX`,).
            ref: Reference pose (px, py, pz, qw, qx, qy, qz). Shape is (7,).
            print_stats_on_failure: Print solver statistics on failure. Default value is `False`.

        Returns:
            Tuple of output state and input sequences respectively. The output state sequence has
            shape (`n_horizon` + 1, 7) and contains camera poses. The output control input sequence
            has shape (`n_horizon + 1`, 6) and contains the true control inputs (camera twist).
        """
        n_horizon = self._ocp.solver_options.N_horizon
        # set initial state and reference
        self._ocp_solver.set(0, "x", x0)
        self._ocp_solver.set(0, "lbx", x0)
        self._ocp_solver.set(0, "ubx", x0)
        self._ocp_solver.set_flat("p", np.tile(ref, n_horizon + 1))
        # solve NLP
        status = self._ocp_solver.solve()
        if status != 0 and print_stats_on_failure:
            self._ocp_solver.print_statistics()
        # get camera pose and twist sequences
        x = self._ocp_solver.get_flat("x").reshape(n_horizon + 1, self.NX)
        return x[:, :7], x[:, 7:]

    def warmup(self, n_shift: int = 1):
        """Warm-up the solver by shifting the previous solution and repeating the last input/state.

        Args:
            n_shift: Shift previous solution by `n_shift` steps and repeat the final inputs/states.
                Should be equal to the number of control inputs that were applied since the last
                `solve` call."""
        n_horizon = self._ocp.solver_options.N_horizon
        assert 0 < n_shift < n_horizon
        # get previous state and input sequences
        x = self._ocp_solver.get_flat("x").reshape(n_horizon + 1, self.NX)
        u = self._ocp_solver.get_flat("u").reshape(n_horizon, self.NU)
        # shift both sequences and repeat them
        x_last, u_last = x[-1], u[-1]
        x = np.roll(x, -n_shift, axis=0)
        x[-n_shift:] = x_last
        u = np.roll(u, -n_shift, axis=0)
        u[-n_shift:] = u_last
        # set state and input sequences
        self._ocp_solver.set_flat("x", x.reshape(-1))
        self._ocp_solver.set_flat("u", u.reshape(-1))

    def get_stats(self, stats: str = "statistics") -> int | float | NDArray:
        """Get solver statistics.

        Args:
            stats: One of the available solver statistics. Valid options include: ['statistics',
                'time_tot', 'time_lin', 'time_sim', 'time_sim_ad', 'time_sim_la', 'time_qp',
                'time_qp_solver_call', 'time_reg', 'time_qpscaling', 'nlp_iter', 'sqp_iter',
                'residuals', 'qp_iter', 'alpha']. Default value is statistics.

        Returns:
            Statistics returned by the underlying OCP solver.
        """
        return self._ocp_solver.get_stats(stats)

    @classmethod
    def _check_dims(cls, cfg: VsOcpSolverCfg) -> None:
        """Check the dimensions of cost matrices and constraint bounds if given."""
        assert cfg.fp.shape == (3,)
        assert cfg.cost_cfg.Q_x.shape == (cls.NX - 1, cls.NX - 1)
        assert cfg.cost_cfg.R_u.shape == (cls.NU, cls.NU)
        assert cfg.cost_cfg.Q_z.shape == (cls.NZ // 2, cls.NZ // 2)
        if cfg.constraint_cfg.lbx.size > 0:
            assert cfg.constraint_cfg.lbx.size == cfg.constraint_cfg.idxbx.size
        if cfg.constraint_cfg.ubx.size > 0:
            assert cfg.constraint_cfg.ubx.size == cfg.constraint_cfg.idxbx.size
        if cfg.constraint_cfg.lbu.size > 0:
            assert cfg.constraint_cfg.lbu.size == cfg.constraint_cfg.idxbu.size
        if cfg.constraint_cfg.ubu.size > 0:
            assert cfg.constraint_cfg.ubu.size == cfg.constraint_cfg.idxbu.size
        if cfg.constraint_cfg.lh.size > 0:
            assert cfg.constraint_cfg.lh.shape == (4 + cls.NZ // 2,)
        if cfg.constraint_cfg.uh.size > 0:
            assert cfg.constraint_cfg.uh.shape == (4 + cls.NZ // 2,)
