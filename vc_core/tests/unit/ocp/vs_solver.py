from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver
from numpy.typing import NDArray

from vc_core.ocp.solver import VsOcpSolver, VsOcpSolverCfg


@pytest.mark.unit
def test_vs_ocp_solver(capsys: pytest.CaptureFixture) -> None:
    # setup solver and integrator
    ocp_solver = setup_ocp_solver()
    integrator = setup_integrator(ocp_solver._ocp)

    # setup arrays for simulation
    Nsim = 100
    nx = ocp_solver.NX
    nu = ocp_solver.NU
    x0 = np.array([0.5, -0.25, -1.25] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 6)
    ref = np.array([-0.1, 0.0, -0.5] + [0.98, 0.04, 0.0087, 0.174])
    ref[3:] /= np.linalg.norm(ref[3:])
    simX = np.zeros((Nsim + 1, nx))
    simX[0, :] = x0
    simU = np.zeros((Nsim, nu))
    t = np.zeros((Nsim))

    # run MPC in closed loop
    ocp_solver.reset(x0)
    with capsys.disabled():
        for i in range(Nsim):
            # solve ocp and get next control input
            ocp_solver.solve(simX[i, :], ref, print_stats_on_failure=True)
            simU[i, :] = ocp_solver.solver.get(0, "u")
            t[i] = ocp_solver.get_stats("time_tot")
            ocp_solver.warmup(n_shift=1)

            # simulate system
            simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

        # evaluate timings (ms)
        t *= 1000
        print(
            f"Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}"
        )

    # plot simulation results
    plot_position(simX[:, :3], ref[:3])
    plot_quaternion(simX[:, 3:7], ref[3:7])
    plot_twist(simX[:, 7:], np.zeros(6))


def setup_ocp_solver() -> VsOcpSolver:
    cfg = VsOcpSolverCfg(
        cost_cfg=VsOcpSolverCfg.CostCfg(Q_x=np.diag([1] * 6 + [0.5] * 6), R_u=np.diag([0.1] * 6)),
        alpha=0.001,
        fp=np.array([0.05, 0.05, 0.0]),
        constraint_cfg=VsOcpSolverCfg.ConstraintCfg(
            lh=np.array([-1.0, -1.0, -1.0, -1.0, -0.5, -0.35]),
            uh=np.array([0.25, 0.25, 1.0, 1.0, 0.5, 0.35]),
        ),
        solver_cfg=VsOcpSolverCfg.SolverCfg(n_horizon=40, time_step=0.1),
    )
    return VsOcpSolver(cfg)


def setup_integrator(ocp: AcadosOcp) -> AcadosSimSolver:
    sim = AcadosSim.from_ocp(ocp)

    sim.solver_options.num_steps = (
        2  # make integrator more precise than the integrator used within the OCP
    )
    sim.code_gen_opts.code_export_directory = "c_generated_code_sim"
    sim.parameter_values = np.zeros(7)
    integrator = AcadosSimSolver(sim)
    return integrator


def plot_position(p: NDArray, r: NDArray) -> None:
    path = Path(__file__).parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    N = p.shape[0]
    steps = np.arange(N)

    _, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    ylabels = ["X (m)", "Y (m)", "Z (m)"]
    for i in range(3):
        axes[i].plot(steps, p[:, i], label="True", color="tab:blue")
        axes[i].plot([0, N - 1], [r[i], r[i]], "r--", label="Reference")
        axes[i].set_ylabel(ylabels[i])
        axes[i].set_xlabel("Step")
        axes[i].grid(True)
        axes[i].legend()
    plt.savefig(path / "position.png", dpi=300)
    plt.close()


def plot_quaternion(q: NDArray, r: NDArray) -> None:
    path = Path(__file__).parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    N = q.shape[0]
    steps = np.arange(N)

    _, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    ylabels = ["W (scalar)", "X (vector)", "Y (vector)", "Z (vector)"]
    for i in range(4):
        axes[i].plot(steps, q[:, i], label="True", color="tab:orange")
        axes[i].plot([0, N - 1], [r[i], r[i]], "r--", label="Reference")
        axes[i].set_ylabel(ylabels[i])
        axes[i].set_xlabel("Step")
        axes[i].grid(True)
        axes[i].legend()
    plt.savefig(path / "orientation.png", dpi=300)
    plt.close()


def plot_twist(v: NDArray, r: NDArray) -> None:
    path = Path(__file__).parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    N = v.shape[0]
    steps = np.arange(N)

    _, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    ylabels = ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)", "Wx (rad/s)", "Wy (rad/s)", "Wz (rad/s)"]
    for i in range(6):
        axes[i].plot(steps, v[:, i], label="True", color="tab:green")
        axes[i].plot([0, N - 1], [r[i], r[i]], "r--", label="Reference")
        axes[i].hlines([-0.25, 0.25], 0, N - 1, linestyles="dashed", label="Limits")
        axes[i].set_ylabel(ylabels[i])
        axes[i].set_xlabel("Step")
        axes[i].grid(True)
        axes[i].legend()
    plt.savefig(path / "twist.png", dpi=300)
    plt.close()
