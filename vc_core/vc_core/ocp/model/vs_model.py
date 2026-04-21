from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
from acados_template import AcadosModel

from .features import feature_dot, project
from .geometry import quat_apply, quat_diff, quat_dot

if TYPE_CHECKING:
    from numpy.typing import NDArray


def export_vs_ode_model(fp: NDArray, alpha: float = 0.001) -> AcadosModel:
    """Defines an instance of `acados_template.AcadosModel` for task-space visual servoing.

    The model's state consists of the following three groups of terms:

    1) `p` **(n=3):** Camera's position wrt reference frame of the visual marker of interest.
    2) `q` **(n=4):** Camera's orientation (scalar-first) wrt reference frame of the visual marker
    of interest.
    3) `u` **(n=6):** Camera's twist vector. The actual control input.

    In addition to the main state, a derived visual state `z` **(n=4)** augments the model's cost
    and constraint formulations. The visual state is made up of the 2D feature location `s` and
    its rate of change `s_dot`.

    The model's control input `u_dot` **(n=6)** represents camera's local acceleration vector.

    The model's "dynamics" are defined through the exponential map of the **SE3** Lie group for the
    camera's pose. The camera twist vector `u` is updated directly through its rate of change `u_dot`.

    The model's stage cost in made up of the three groups of terms corresponding to state `x`,
    input `u_dot` and visual feature velocities `s_dot`.

    The model's constraints are made up the traditional state `x` and input `u_dot` box contraints
    along with nonlinear visual feature `s` visibility constraints.

    Args:
        fp: Feature coordinates wrt to the reference frame of the visual marker of interest.
        alpha: Coefficient of quat norm restorative term in equation for `qdot`. This prevents the
            quaternion's magnitude from drifting away from 1 during integration over the full
            prediction horizon. Default value is 0.001.

    Returns:
        Acados model derived from the above formulation.
    """
    # setup state
    p = ca.SX.sym("p", 3)
    q = ca.SX.sym("q", 4)
    u = ca.SX.sym("u", 6)
    x = ca.vertcat(p, q, u)

    # setup visual features
    fp = ca.SX([fp[0], fp[1], fp[2]])
    s, depth = project(p, q, fp)
    s_dot = feature_dot(u, s, depth)

    # setup input
    u_dot = ca.SX.sym("u_dot", 6)

    # setup reference
    p_ref = ca.SX.sym("p_ref", 3)
    q_ref = ca.SX.sym("q_ref", 4)
    ref = ca.vertcat(p_ref, q_ref)

    # setup xdot
    xdot = ca.SX.sym("xdot", x.size()[0])

    # setup dynamics
    v, w = u[0:3], u[3:6]
    p_dot = quat_apply(q, v)
    q_dot = quat_dot(q, w) + alpha * (1.0 - ca.dot(q, q)) * q
    f_expl = ca.vertcat(p_dot, q_dot, u_dot)
    f_impl = xdot - f_expl

    # setup costs
    p_err = p - p_ref
    q_err = quat_diff(q, q_ref)
    cost_y = ca.vertcat(p_err, q_err, u, u_dot, s_dot)
    cost_y_e = ca.vertcat(p_err, q_err, u)

    # setup nonlinear constraints
    con_h_expr = s

    # setup model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u_dot
    model.p = ref
    model.cost_y_expr = cost_y
    model.cost_y_expr_e = cost_y_e
    model.con_h_expr = con_h_expr

    # setup model meta information
    model.name = "vs_ode"
    model.x_labels = ["$p$ [m]", "$q$ [rad]", "$v$ [m/s]", r"$\omega$ [rad/s]"]
    model.u_labels = [r"$\dot{v}$ [m/s^2]", r"$\dot{\omega}$ [rad/s^2]"]
    model.t_label = "$t$ [s]"
    return model
