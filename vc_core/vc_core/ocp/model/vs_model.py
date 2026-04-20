import casadi as ca
from acados_template import AcadosModel

from .geometry import quat_apply, quat_dot


def export_vs_ode_model(alpha: float = 1.0) -> AcadosModel:
    """Defines an instance of `acados_template.AcadosModel` for task-space visual servoing.

    The model's state consists of the following three groups of terms:

    1) `p` **(n=3):** Camera's position wrt reference frame of a visual marker of interest.
    2) `q` **(n=4):** Camera's orientation (scalar-first) wrt reference frame of a visual marker of interest.
    4) `u` **(n=6):** Camera's twist vector. The actual control input.

    The model's control input `u_dot` **(n=6)** represents camera's local acceleration vector.

    The model's "dynamics" are defined through the exponential map of the **SE3** Lie group for the
    camera's pose. The camera twist vector `u` is updated directly through its rate of change `u_dot`.

    Args:
        alpha: Coefficient of quat norm restorative term in equation for `qdot`. This prevents the
            quaternion's magnitude from drifting away from 1 during integration over the full
            prediction horizon. Default value is 1.0.

    Returns:
        Acados model derived from the above formulation.
    """
    # setup state
    p = ca.SX.sym("p", 3)
    q = ca.SX.sym("q", 4)
    u = ca.SX.sym("u", 6)
    x = ca.vertcat(p, q, u)

    # setup input
    u_dot = ca.SX.sym("u", 6)

    # setup xdot
    v, w = u[0:3], u[3:6]
    p_dot = quat_apply(q, v)
    q_dot = quat_dot(q, w) + alpha * (1.0 - ca.dot(q, q)) * q
    xdot = ca.vertcat(p_dot, q_dot, u_dot)

    # dynamics
    f_expl = xdot
    f_impl = xdot - f_expl

    # setup model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u

    # setup model meta information
    model.name = "vs_ode"
    model.x_labels = ["$p$ [m]", "$q$ [rad]", "$v$ [m/s]", r"$\omega$ [rad/s]"]
    model.u_labels = [r"$\dot{v}$ [m/s^2]", r"$\dot{\omega}$ [rad/s^2]"]
    model.t_label = "$t$ [s]"
    return model
