# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .common import CONTACT_FORCE_HIGH_THRESHOLD, CONTACT_FORCE_LOW_THRESHOLD, contact_force_magnitudes


def finger_state_normalized_from_open(
    env,
    asset_cfg: SceneEntityCfg,
    lower_rad: float,
    upper_rad: float,
) -> torch.Tensor:
    """Return normalized per-finger state in [0, 1] from selected joint positions.

    Intended usage: select only `joint4` for each finger via `asset_cfg.joint_ids` and pass
    `lower_rad=-4°` and `upper_rad=30°` (in radians). The output shape is
    `(num_envs, num_selected_joints)`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    denom = float(upper_rad - lower_rad)
    if abs(denom) < 1e-8:
        raise ValueError("`upper_rad` must be different from `lower_rad`.")

    normalized_joints = (joint_pos - float(lower_rad)) / denom
    return normalized_joints.clamp(0.0, 1.0)


def contact_forces_from_sensors(
    env,
    sensor_names: list[str],
) -> torch.Tensor:
    """Return per-sensor contact-force magnitudes for each named sensor.

    The returned Tensor has shape (num_envs, num_sensors) with the
    world-frame net force magnitudes for each sensor (i.e. ||f||).
    """
    return contact_force_magnitudes(env, sensor_names)


def net_vs_sphere_contact_force_magnitude_diff(
    env,
    sensor_names: list[str],
) -> torch.Tensor:
    """Return per-sensor absolute difference between net and sphere-contact force magnitudes.

    For each named sensor this computes:
    ``| ||net_forces_w|| - ||force_matrix_w(sphere)|| |``.

    The returned Tensor has shape ``(num_envs, num_sensors)``.
    """
    sensors = [env.scene.sensors[name] for name in sensor_names]

    net_force_magnitudes = torch.stack(
        [torch.linalg.vector_norm(sensor.data.net_forces_w.view(env.num_envs, -1, 3).sum(dim=1), dim=-1) for sensor in sensors],
        dim=1,
    )
    sphere_force_magnitudes = torch.stack(
        [
            torch.linalg.vector_norm(sensor.data.force_matrix_w.view(env.num_envs, -1, 3).sum(dim=1), dim=-1)
            for sensor in sensors
        ],
        dim=1,
    )
    return torch.abs(net_force_magnitudes - sphere_force_magnitudes)


def contact_force_flags(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
) -> torch.Tensor:
    """Return per-sensor boolean flags (as float32) indicating whether the contact force
    exceeds the high threshold and whether it is below the low threshold.

    For N sensors this returns a Tensor shaped (num_envs, N * 2) with per-sensor flags in the
    order [s1_exceeds, s1_below, s2_exceeds, s2_below, ...].
    """
    mag = contact_force_magnitudes(env, sensor_names)
    exceeds = (mag > high).to(torch.float32)
    below = (mag < low).to(torch.float32)
    flags = torch.stack((exceeds, below), dim=-1)
    return flags.reshape(flags.shape[0], -1)


def contact_force_mid_diff(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
) -> torch.Tensor:
    """Return the signed difference between the midpoint of [low, high] and the
    current contact-force magnitudes for each named sensor.

    The returned Tensor has shape (num_envs, num_sensors) with values `mid - mag`.
    Positive values indicate the force is below the midpoint, negative values indicate
    the force is above the midpoint.
    """
    mag = contact_force_magnitudes(env, sensor_names)
    mid = (low + high) / 2.0
    return mid - mag


def sphere_size(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
    """Return cached sphere size (base radius * applied scale)."""
    return env._three_fingers_sphere_size_cache


def sphere_diameter(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
    """Return cached sphere diameter in meters."""
    return 2.0 * sphere_size(env, asset_cfg)


# relative fingertip-to-sphere vector ------------------------------------------------

def link5_to_sphere_pos(
    env,
    fingers_asset_cfg: SceneEntityCfg,
    sphere_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flattened vectors from each fingertip (link5) to the sphere centre.

    Only the current timestep is returned; any temporal history is handled by
    the observation manager (``history_length`` field of ``ObsTerm``).
    """
    # fetch fingertip world poses directly from scene data
    asset = env.scene[fingers_asset_cfg.name]
    # the original ``body_pos_w`` helper subtracts ``env_origins``; replicate
    pos = asset.data.body_pos_w[:, fingers_asset_cfg.body_ids, :3]
    pos = pos - env.scene.env_origins.unsqueeze(1)
    # pos shape: [N, B, 3]
    N = env.num_envs
    B = pos.shape[1]
    pos_flat = pos.view(N, B * 3)

    sph = env.scene[sphere_asset_cfg.name].data.root_pos_w - env.scene.env_origins
    sph_flat = sph.view(N, 3).repeat(1, B)

    return pos_flat - sph_flat
