from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ============================================================
# Parameters
# ============================================================

N = 6  # Number of sampled poses
offset_distance = 0.25  # Distance above marker [m]
theta_deg = 25.0  # Rotation magnitude [deg]

theta = np.deg2rad(theta_deg)
step = 2.0 * np.pi / N

# ============================================================
# Utility Functions
# ============================================================


def make_transform(rmat: NDArray, tvec: NDArray) -> NDArray:
    """Construct homogeneous transform."""
    tf = np.eye(4)
    tf[:3, :3] = rmat
    tf[:3, 3] = tvec
    return tf


def transform_points(tf: NDArray, pts: NDArray) -> NDArray:
    """Apply homogeneous transform to Nx3 points."""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    transformed = (tf @ pts_h.T).T
    return transformed[:, :3]


def create_camera_frustum(scale=0.02) -> tuple[NDArray, list[list[int]]]:
    """Simple pyramid camera frustum. Returns Nx3 vertices and edge connectivity."""
    pts = (
        np.array(
            [
                [0, 0, 0],  # camera center
                [-1, -1, 2],
                [1, -1, 2],
                [1, 1, 2],
                [-1, 1, 2],
            ]
        )
        * scale
    )

    edges = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    return pts, edges


def add_frame(plotter: pv.Plotter, tf: NDArray, scale: float = 0.03) -> None:
    """Add coordinate frame visualization."""
    origin = tf[:3, 3]

    x_axis = origin + tf[:3, 0] * scale
    y_axis = origin + tf[:3, 1] * scale
    z_axis = origin + tf[:3, 2] * scale

    plotter.add_lines(np.vstack([origin, x_axis]), color="red", width=3)
    plotter.add_lines(np.vstack([origin, y_axis]), color="green", width=3)
    plotter.add_lines(np.vstack([origin, z_axis]), color="blue", width=3)


def add_camera(
    plotter: pv.Plotter,
    tf: NDArray,
    scale: float = 0.02,
    color: str = "white",
    label: str | None = None,
) -> None:
    """Add camera frustum."""
    pts, edges = create_camera_frustum(scale)
    transformed_pts = transform_points(tf, pts)
    for i, e in enumerate(edges):
        line = transformed_pts[e]
        label = label if i == 0 else None
        plotter.add_lines(line, color=color, width=2, label=label)


# ============================================================
# Create Initial Pose
# ============================================================

# Marker frame:
# XY plane = board plane
# +Z = board normal

# Camera initially placed above marker
# Camera Z-axis points downward toward marker

R_init = R.from_euler("x", 180, degrees=True).as_matrix()
t_init = np.array([0, 0, offset_distance])
T_init = make_transform(R_init, t_init)

# ============================================================
# Sample Rotation Axes
# ============================================================

rotation_axes = []
for i in range(N):
    axis = np.array([np.cos(i * step), np.sin(i * step), 0.0])
    rotation_axes.append(axis)

# ============================================================
# Generate Sampled Camera Poses
# ============================================================

sampled_poses = []
for axis in rotation_axes:
    rotvec = axis * theta
    R_offset = R.from_rotvec(rotvec).as_matrix()
    # Pre-rotation
    R_sample = R_offset @ R_init
    t_sample = R_offset @ t_init
    # Make transforms
    T_sample = make_transform(R_sample, t_sample)
    sampled_poses.append(T_sample)

# ============================================================
# Visualization
# ============================================================

plotter = pv.Plotter(window_size=(1400, 1000))

# ------------------------------------------------------------
# Calibration Board
# ------------------------------------------------------------

board_size = 0.2
plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=board_size, j_size=board_size)
plotter.add_mesh(plane, color="gray", opacity=0.9, show_edges=True, label=r"Calibration Board")

# ------------------------------------------------------------
# Sphere Visualization
# ------------------------------------------------------------

sphere = pv.Sphere(radius=offset_distance, center=(0, 0, 0))
plotter.add_mesh(sphere, color="cyan", opacity=0.05)

# ------------------------------------------------------------
# Marker Coordinate Frame
# ------------------------------------------------------------

T_marker = make_transform(R_init, np.zeros(3))
add_frame(plotter, T_marker, scale=0.05)

# ------------------------------------------------------------
# Initial Camera Pose
# ------------------------------------------------------------

add_camera(plotter, T_init, scale=0.025, color="green", label=r"Reference Pose")
add_frame(plotter, T_init, scale=0.04)

# ------------------------------------------------------------
# Sampled Poses
# ------------------------------------------------------------

for i, T in enumerate(sampled_poses):
    label = r"Sampled Poses" if i == 0 else None
    add_camera(plotter, T, scale=0.0125, color="lime", label=label)
    add_frame(plotter, T, scale=0.025)

# ============================================================
# Theta Arc Visualization
# ============================================================

# Choose one sampled pose for visualization
p0 = T_init[:3, 3]
axis = rotation_axes[-1]

# Create spherical interpolation arc
num_arc_pts = 100
arc_pts = []
for t in np.linspace(0.0, 1.0, num_arc_pts):
    # Slerp-like interpolation on sphere
    angle = theta * t
    R_interp = R.from_rotvec(axis * angle).as_matrix()
    p_interp = R_interp @ p0
    arc_pts.append(p_interp)
arc_pts = np.array(arc_pts)

theta_arc = pv.Spline(arc_pts, 300)
plotter.add_mesh(theta_arc, color="red", line_width=5, opacity=0.6)

# ============================================================
# Distance d Visualization
# ============================================================

board_center = np.array([0.0, 0.0, 0.0])
camera_center = T_init[:3, 3]

# Arrow from board to nominal pose
d_arrow = pv.Arrow(
    start=board_center,
    direction=(camera_center - board_center),
    tip_radius=0.025,
    shaft_radius=0.01,
    scale="auto",
)
plotter.add_mesh(d_arrow, color="gray", opacity=0.4)

# ------------------------------------------------------------
# Camera Trajectory Arc
# ------------------------------------------------------------

sampled_positions = np.array([T[:3, 3] for T in sampled_poses])
trajectory = pv.Spline(sampled_positions, 400, closed=True)
plotter.add_mesh(trajectory, color="magenta", line_width=4, opacity=0.6)

# ============================================================
# Render Settings
# ============================================================

plotter.set_background("white")
plotter.camera_position = [
    (0.36, -0.406, 0.664),  # camera position
    (-0.052, 0.031, 0.050),  # focal point
    (-0.419, 0.583, 0.696),  # up vector
]

path = Path(__file__).parent / "figures"
path.mkdir(parents=True, exist_ok=True)
plotter.add_legend(border=True, bcolor=(0.9, 0.9, 0.9), face="circle", loc="upper left")
plotter.save_graphic(path / "he_samples.svg")
plotter.show()

print(f"Final camera position: {plotter.camera_position}")
