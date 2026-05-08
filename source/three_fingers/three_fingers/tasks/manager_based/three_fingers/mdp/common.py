# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

# Contact force thresholds for grasping
CONTACT_FORCE_LOW_THRESHOLD = 1.35#1.75  # Minimum force to consider proper contact  - forces should be from 1.75 to 2.1 for stable grasping
CONTACT_FORCE_HIGH_THRESHOLD = 2.5#2.1  # Maximum force before excessive pressure
# CRITICAL threshold beyond which a contact force is considered damaging and should terminate the episode.
CONTACT_FORCE_CRITICAL_THRESHOLD = 2.5

# Active-joint effort magnitude thresholds for stable grasping
JOINT_EFFORT_LOW_THRESHOLD = 0.23
JOINT_EFFORT_HIGH_THRESHOLD = 0.35


def contact_force_vectors(env, sensor_names: list[str]) -> torch.Tensor:
    """Return per-sensor world-frame contact-force vectors from ``force_matrix_w``.

    For each sensor, this sums the contact-force matrix over body/filter entries so the
    output represents only the configured filtered contacts (sphere↔link in this env).

    Returns shape ``(num_envs, num_sensors, 3)``.
    """
    return torch.stack(
        [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, -1, 3).sum(dim=1) for name in sensor_names],
        dim=1,
    )


def contact_force_magnitudes(env, sensor_names: list[str]) -> torch.Tensor:
    """Return per-sensor contact-force magnitudes with shape (num_envs, num_sensors)."""
    forces = contact_force_vectors(env, sensor_names)
    return torch.linalg.vector_norm(forces, dim=-1)


def tanh_shaped_positive_reward(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Map a non-negative error-like tensor x -> reward in (0, 1],
    high reward when x is near zero: reward = 1 - tanh(scale * x).
    """
    return (1.0 - torch.tanh(scale * x)).clamp(min=0.0, max=1.0)
