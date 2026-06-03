# Visual Control Core Package

## Package Details

Provides the reusable backends and shared functionality used throughout the repository. This package contains the majority of algorithmic implementations and serves as the foundation for perception, estimation, planning, and experimentation.
The package serves as the common algorithmic foundation used by all higher-level packages.

## Core Components

### C++ Components

* Extended Kalman Filter implementations
* Low-Pass Filter implementations
* Visual servoing backends
* Inverse kinematics solvers
* Shared utilities

#### Python Components

* NVIDIA Kaolin differentiable rendering backend
* PyTorch3D differentiable rendering backend
* Differentiable rendering optimization models
* PyTorch-based optimizers and training utilities
* Objective functions and evaluation metrics
* CasADi optimal control problem definitions
* acados integrations
* SAM2 image and live-video segmentation interfaces
* Plotting and visualization utilities
* Experiment loggers (Console, CSV, WandB)