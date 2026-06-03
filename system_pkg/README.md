# System Package

## Package Details

Integrates all repository packages into complete operational workflows through ROS 2 launch files and system-level orchestration. This package serves as the primary entry point for users of the framework.
The package is responsible for coordinating package interactions, managing launch configurations, and defining complete system workflows.

## Operational Modes

### Main Operation

Launches the complete perception, estimation, planning, and control pipeline for standard system operation.

### Hand-Eye Calibration

Executes the hand-eye calibration workflow and terminates once calibration has completed successfully.

### Eye Calibration

Collects observations from multiple viewpoints and performs joint optimization of:

* Eye geometry
* Eye texture
* Camera parameters
* Multi-view consistency constraints

using differentiable rendering and optimization-based estimation techniques, then stores the final optimized geometry and texture as well as additional figures for debugging.
