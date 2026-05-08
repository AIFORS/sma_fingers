# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Sequence, cast

import torch
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

from .common import (
    CONTACT_FORCE_CRITICAL_THRESHOLD,
    CONTACT_FORCE_HIGH_THRESHOLD,
    CONTACT_FORCE_LOW_THRESHOLD,
    contact_force_vectors,
    contact_force_magnitudes,
    tanh_shaped_positive_reward,
)


def contact_force_exceeds_critical(
    env,
    sensor_names: list[str],
    critical: float = CONTACT_FORCE_CRITICAL_THRESHOLD,
) -> torch.Tensor:
    """Return True for environments where ANY sensor contact force magnitude exceeds `critical`."""
    mag = contact_force_magnitudes(env, sensor_names)
    exceeds_critical = mag > critical
    return torch.any(exceeds_critical, dim=1)


def _grasp_success_mask(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
    sphere_asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    fingers_asset_cfg: SceneEntityCfg | None = None,
    lin_vel_thresh: float = 0.01,
    ang_vel_thresh: float = 0.01,
    joint_vel_thresh: float = 0.01,
    action_tol: float = 0.02,
    joint_symmetry_tol: float | None = None,
    require_sphere_z_above_radius: bool = False,
) -> torch.Tensor:
    """Return a boolean mask of environments satisfying all conditions for a successful grasp."""
    mag = contact_force_magnitudes(env, sensor_names)
    in_range = (mag >= low) & (mag <= high)
    contact_ok = in_range.all(dim=1)

    sphere: RigidObject = env.scene[sphere_asset_cfg.name]
    lin_speed = torch.linalg.vector_norm(sphere.data.root_lin_vel_w, dim=1)
    ang_speed = torch.linalg.vector_norm(sphere.data.root_ang_vel_w, dim=1)
    sphere_static = (lin_speed <= lin_vel_thresh) & (ang_speed <= ang_vel_thresh)

    if fingers_asset_cfg is None:
        raise ValueError("fingers_asset_cfg must be provided to check finger motion and action closeness.")
    fingers = env.scene[fingers_asset_cfg.name]
    joint_vel = fingers.data.joint_vel[:, fingers_asset_cfg.joint_ids]
    fingers_static = (torch.abs(joint_vel) <= joint_vel_thresh).all(dim=1)

    processed = env.action_manager.get_term("finger_targets").processed_actions
    joint_pos = fingers.data.joint_pos[:, fingers_asset_cfg.joint_ids]
    action_close = (torch.abs(processed - joint_pos) <= action_tol).all(dim=1)

    if joint_symmetry_tol is not None:
        pairwise_diff = torch.abs(processed.unsqueeze(2) - processed.unsqueeze(1))
        action_symmetric = (pairwise_diff <= joint_symmetry_tol).all(dim=(1, 2))
    else:
        action_symmetric = torch.ones_like(contact_ok)

    if require_sphere_z_above_radius:
        pos_z = sphere.data.root_pos_w[:, 2]
        radius_cache = getattr(env, "_three_fingers_sphere_size_cache", None)
        if radius_cache is None:
            base_radius = float(getattr(sphere.cfg.spawn, "radius", 0.0))
            radius = torch.full_like(pos_z, base_radius)
        else:
            radius = radius_cache.view(-1)
        sphere_height_ok = pos_z > radius
    else:
        sphere_height_ok = torch.ones_like(contact_ok)

    return contact_ok & sphere_static & fingers_static & action_close & action_symmetric & sphere_height_ok


def grasp_success(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
    sphere_asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    fingers_asset_cfg: SceneEntityCfg | None = None,
    lin_vel_thresh: float = 0.01,
    ang_vel_thresh: float = 0.01,
    joint_vel_thresh: float = 0.01,
    action_tol: float = 0.02,
    joint_symmetry_tol: float | None = None,
    require_sphere_z_above_radius: bool = False,
) -> torch.Tensor:
    """Termination term: True where a successful grasp is detected."""
    return _grasp_success_mask(
        env,
        sensor_names=sensor_names,
        low=low,
        high=high,
        sphere_asset_cfg=sphere_asset_cfg,
        fingers_asset_cfg=fingers_asset_cfg,
        lin_vel_thresh=lin_vel_thresh,
        ang_vel_thresh=ang_vel_thresh,
        joint_vel_thresh=joint_vel_thresh,
        action_tol=action_tol,
        joint_symmetry_tol=joint_symmetry_tol,
        require_sphere_z_above_radius=require_sphere_z_above_radius,
    )


class GraspSuccessWithActionConstancy(ManagerTermBase):
    """Stateful grasp-success term gated by action constancy over a short history."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cast(dict[str, object], cfg.params)

        action_name = params.get("action_name", "finger_targets")
        self.action_name = action_name if isinstance(action_name, str) else "finger_targets"

        action_constancy_steps = params.get("action_constancy_steps", 10)
        if isinstance(action_constancy_steps, (int, float, str)):
            self.action_constancy_steps = max(1, int(action_constancy_steps))
        else:
            self.action_constancy_steps = 10

        action_constancy_tol = params.get("action_constancy_tol", 0.01)
        if isinstance(action_constancy_tol, (int, float, str)):
            self.action_constancy_tol = max(0.0, float(action_constancy_tol))
        else:
            self.action_constancy_tol = 0.01
        self._action_history: torch.Tensor | None = None
        self._history_fill: torch.Tensor = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._history_write_index: torch.Tensor = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        if self._action_history is None:
            return
        if env_ids is None:
            self._action_history[:] = 0.0
            self._history_fill[:] = 0
            self._history_write_index[:] = 0
        else:
            self._action_history[env_ids, :, :] = 0.0
            self._history_fill[env_ids] = 0
            self._history_write_index[env_ids] = 0

    def __call__(
        self,
        env,
        sensor_names: list[str],
        low: float = CONTACT_FORCE_LOW_THRESHOLD,
        high: float = CONTACT_FORCE_HIGH_THRESHOLD,
        sphere_asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
        fingers_asset_cfg: SceneEntityCfg | None = None,
        lin_vel_thresh: float = 0.01,
        ang_vel_thresh: float = 0.01,
        joint_vel_thresh: float = 0.01,
        action_tol: float = 0.02,
        joint_symmetry_tol: float | None = None,
        require_sphere_z_above_radius: bool = False,
        action_name: str | None = None,
        action_constancy_steps: int | None = None,
        action_constancy_tol: float | None = None,
        return_float: bool = False,
    ) -> torch.Tensor:
        action_term_name = self.action_name if action_name is None else action_name
        window_steps = self.action_constancy_steps if action_constancy_steps is None else max(1, int(action_constancy_steps))
        tol = self.action_constancy_tol if action_constancy_tol is None else max(0.0, float(action_constancy_tol))

        processed = env.action_manager.get_term(action_term_name).processed_actions
        num_envs, action_dim = processed.shape

        if (
            self._action_history is None
            or self._action_history.shape[0] != num_envs
            or self._action_history.shape[2] != action_dim
            or self._action_history.shape[1] != window_steps
            or self._action_history.device != processed.device
        ):
            self._action_history = torch.zeros((num_envs, window_steps, action_dim), device=processed.device)
            self._history_fill = torch.zeros(num_envs, dtype=torch.long, device=processed.device)
            self._history_write_index = torch.zeros(num_envs, dtype=torch.long, device=processed.device)

        env_ids = torch.arange(num_envs, device=processed.device)
        write_index = self._history_write_index
        self._action_history[env_ids, write_index, :] = processed
        self._history_write_index = (write_index + 1) % window_steps
        self._history_fill = torch.clamp(self._history_fill + 1, max=window_steps)

        history_ready = self._history_fill == window_steps
        action_span = self._action_history.max(dim=1).values - self._action_history.min(dim=1).values
        stable = (action_span <= tol).all(dim=1)
        action_constancy_ok = history_ready & stable

        success_mask = _grasp_success_mask(
            env,
            sensor_names=sensor_names,
            low=low,
            high=high,
            sphere_asset_cfg=sphere_asset_cfg,
            fingers_asset_cfg=fingers_asset_cfg,
            lin_vel_thresh=lin_vel_thresh,
            ang_vel_thresh=ang_vel_thresh,
            joint_vel_thresh=joint_vel_thresh,
            action_tol=action_tol,
            joint_symmetry_tol=joint_symmetry_tol,
            require_sphere_z_above_radius=require_sphere_z_above_radius,
        )
        final_mask = success_mask & action_constancy_ok
        return final_mask.to(torch.float32) if return_float else final_mask


def sphere_out_of_xy_bounds(
    env,
    radius: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
) -> torch.Tensor:
    """Terminate when the asset's root x,y distance from the world origin exceeds `radius`."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    env_origins = env.scene.env_origins[:, :2]
    return torch.linalg.vector_norm(pos_xy - env_origins, dim=1) > radius
