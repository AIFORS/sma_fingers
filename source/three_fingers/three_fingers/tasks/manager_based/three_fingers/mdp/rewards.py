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
    JOINT_EFFORT_HIGH_THRESHOLD,
    JOINT_EFFORT_LOW_THRESHOLD,
    contact_force_vectors,
    contact_force_magnitudes,
    tanh_shaped_positive_reward,
)

# termination helpers are kept in a separate module so they can be reused
from .terminations import _grasp_success_mask


def randomize_rigid_body_scale_cached(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    rand_samples = rand_samples.tolist()

    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    sizes = torch.empty(env.scene.num_envs, device=env.device, dtype=torch.float32)
    base_radius = getattr(asset.cfg.spawn, "radius", 1.0)

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            sizes[env_id] = base_radius * (rand_samples[i][0] + rand_samples[i][1] + rand_samples[i][2]) / 3.0

    setattr(env, "_three_fingers_sphere_size_cache", sizes.view(-1, 1))


def reset_sphere_state(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    spawn_cap: float = 0.065,
):
    asset: RigidObject = env.scene[asset_cfg.name]
    env_idx = env_ids.long()
    theta = torch.rand((env_idx.numel(), 1), device=asset.device) * (2 * math.pi)
    reach = torch.sqrt(torch.rand((env_idx.numel(), 1), device=asset.device)) * spawn_cap
    pos_xy = torch.cat((torch.cos(theta), torch.sin(theta)), dim=1) * reach
    root = asset.data.default_root_state[env_idx].clone()
    root[:, :2] = env.scene.env_origins[env_idx, :2] + pos_xy
    root[:, 7:] = 0.0
    asset.write_root_state_to_sim(root, env_ids=cast(Sequence[int], env_idx))


def finger_symmetry(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.abs(joint_pos - joint_pos.mean(dim=1, keepdim=True)), dim=1)


def contact_force_balance(env, sensor_names: list[str]) -> torch.Tensor:
    """Return a normalized *similarity* score in [0, 1] where 1.0 means all
    sensors have identical contact-force magnitudes and 0.0 means large
    imbalance.

    The previous implementation returned an unbounded mean absolute deviation
    (penalty). This version computes the mean absolute deviation normalized by
    the mean magnitude (coefficient-of-variation-like), then converts it into a
    similarity score: similarity = clamp(1 - relative_deviation, 0, 1).

    This makes the term suitable to use directly as a reward (higher is
    better) and is scale-invariant with respect to absolute force magnitudes.
    """
    mag = contact_force_magnitudes(env, sensor_names)

    mean_mag = mag.mean(dim=1)
    mad = torch.mean(torch.abs(mag - mean_mag.unsqueeze(1)), dim=1)
    rel_dev = mad / (mean_mag + 1e-6)

    similarity = (1.0 - rel_dev).clamp(min=0.0, max=1.0)
    return similarity


def is_touching(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
) -> torch.Tensor:
    """Return 1.0 where all finger contact force magnitudes are in [low, high], else 0.0."""
    mag = contact_force_magnitudes(env, sensor_names)
    in_range = (mag > 0) & (mag <= high)
    return in_range.all(dim=1).to(torch.float32)


def contact_force_in_range(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
) -> torch.Tensor:
    """Return a dense reward in [0, 1] that peaks near the middle of [low, high]."""
    mag = contact_force_magnitudes(env, sensor_names)

    mid = (low + high) / 2.0
    half = (high - low) / 2.0

    t = (mag - mid) / half
    per_sensor = tanh_shaped_positive_reward(torch.abs(t), scale=10.0)

    in_interval = (mag >= low) & (mag <= high)
    per_sensor = torch.where(in_interval, per_sensor, torch.zeros_like(per_sensor))
    per_sensor = per_sensor.clamp(min=0.0, max=1.0)

    return torch.mean(per_sensor, dim=1)


def contact_force_exceeds(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
) -> torch.Tensor:
    """Return the total amount by which sensors' contact force magnitudes exceed high."""
    mag = contact_force_magnitudes(env, sensor_names)
    above = (mag - high).clamp(min=0.0)
    deviation = above
    return torch.sum(deviation, dim=1)


def joint_effort_all_in_range(
    env,
    asset_cfg: SceneEntityCfg,
    low: float = JOINT_EFFORT_LOW_THRESHOLD,
    high: float = JOINT_EFFORT_HIGH_THRESHOLD,
) -> torch.Tensor:
    """Return 1.0 where all selected joint effort magnitudes are in [low, high], else 0.0."""
    asset: Articulation = env.scene[asset_cfg.name]
    effort_mag = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    in_range = (effort_mag >= low) & (effort_mag <= high)
    return in_range.all(dim=1).to(torch.float32)




def finger_contact_penalty(
    env,
    sensor_names: list[str],
    touch_threshold: float = 0.0,
    force_scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for inter-finger contact based on count and force magnitude."""
    mag = contact_force_magnitudes(env, sensor_names)
    excess = (mag - touch_threshold).clamp(min=0.0)
    touch_count = (excess > 0.0).to(torch.float32).sum(dim=1)
    force_sum = excess.sum(dim=1)
    return touch_count + force_scale * force_sum






def grasp_success_reward(
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
) -> torch.Tensor:
    """Reward term that returns 1.0 when grasp success is achieved, else 0.0."""
    mask = _grasp_success_mask(
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
    )
    return mask.to(torch.float32)




def opening_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty proportional to the amount the target command would open the finger."""
    targets = env.action_manager.get_term("finger_targets").processed_actions
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    diff = (joint_pos - targets).clamp(min=0.0)
    return torch.sum(diff, dim=1)


def action_deviation(env, asset_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    targets = env.action_manager.get_term("finger_targets").processed_actions
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum((torch.abs(targets - joint_pos) - threshold).clip(min=0.0), dim=1)


def sphere_xy_distance(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
) -> torch.Tensor:
    """Return an exponential penalty based on the L2 distance of the asset's root position in XY."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    env_origins = env.scene.env_origins[:, :2]
    dist = torch.linalg.vector_norm(pos_xy - env_origins, dim=1)
    return torch.exp(dist * 5.0) - 1.0


def sphere_z_above_radius(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
) -> torch.Tensor:
    """Binary reward: 1.0 when the sphere's root z coordinate exceeds its current radius.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_z = asset.data.root_pos_w[:, 2]
    return (pos_z > env._three_fingers_sphere_size_cache.view(-1)).to(torch.float32)




def sphere_lin_acc_mag(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
    """Return the L2 norm of the sphere's linear acceleration (m/s^2) per environment."""
    asset: RigidObject = env.scene[asset_cfg.name]
    acc = asset.data.body_com_acc_w[..., :3].squeeze(1)
    return torch.linalg.vector_norm(acc, dim=1)


def sphere_lin_vel_mag(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
    """Return the L2 norm of the sphere's linear velocity (m/s) per environment."""
    asset: RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_w
    return torch.linalg.vector_norm(vel, dim=1)


def tanh_sphere_lin_vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    scale: float = 50.0,
) -> torch.Tensor:
    """Tanh-shaped reward that favors near-zero sphere linear speed."""
    speed = sphere_lin_vel_mag(env, asset_cfg)
    return tanh_shaped_positive_reward(speed, scale)


def tanh_sphere_lin_acc_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    scale: float = 20.0,
) -> torch.Tensor:
    """Tanh-shaped reward that favors low linear acceleration of the sphere."""
    acc = sphere_lin_acc_mag(env, asset_cfg)
    return tanh_shaped_positive_reward(acc, scale)


def tanh_contact_midpoint_reward(
    env,
    sensor_names: list[str],
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
    scale: float = 3.0,
    require_touch: bool = True,
) -> torch.Tensor:
    """Per-sensor tanh reward encouraging contact-force magnitudes near midpoint of [low, high]."""
    mag = contact_force_magnitudes(env, sensor_names)

    mid = (low + high) / 2.0
    abs_diff = torch.abs(mag - mid)

    per_sensor = tanh_shaped_positive_reward(abs_diff, scale)

    if require_touch:
        touching = (mag >= low).to(per_sensor.dtype)
        per_sensor = per_sensor * touching
        denom = touching.sum(dim=1).clamp(min=1.0)
    else:
        denom = torch.tensor(mag.shape[1], device=mag.device, dtype=per_sensor.dtype)

    per_env = per_sensor.sum(dim=1) / denom
    return per_env


def tanh_contacted_finger_vel_reward(
    env,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg,
    low: float = CONTACT_FORCE_LOW_THRESHOLD,
    high: float = CONTACT_FORCE_HIGH_THRESHOLD,
    scale: float = 20.0,
) -> torch.Tensor:
    """Tanh-shaped reward for low joint velocities when all named contact sensors are in-range."""
    mag = contact_force_magnitudes(env, sensor_names)
    in_range_all = ((mag >= low) & (mag <= high)).all(dim=1)

    fingers = env.scene[asset_cfg.name]
    joint_vel = fingers.data.joint_vel[:, asset_cfg.joint_ids]
    speed = torch.linalg.vector_norm(joint_vel, dim=1)

    reward = tanh_shaped_positive_reward(speed, scale)
    return reward * in_range_all.to(reward.dtype)


class TanhContactConstancyReward(ManagerTermBase):
    """Manager-based reward that encourages in-range contact magnitudes to remain constant."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cast(dict[str, object], cfg.params)

        sensor_names = params.get("sensor_names")
        if not isinstance(sensor_names, list) or not all(isinstance(name, str) for name in sensor_names):
            raise ValueError("'sensor_names' must be provided in params for TanhContactConstancyReward")
        self.sensor_names = list(sensor_names)

        low = params.get("low", CONTACT_FORCE_LOW_THRESHOLD)
        self.low = float(low) if isinstance(low, (int, float, str)) else CONTACT_FORCE_LOW_THRESHOLD

        high = params.get("high", CONTACT_FORCE_HIGH_THRESHOLD)
        self.high = float(high) if isinstance(high, (int, float, str)) else CONTACT_FORCE_HIGH_THRESHOLD

        scale = params.get("scale", 10.0)
        self.scale = float(scale) if isinstance(scale, (int, float, str)) else 10.0

        num_envs = env.num_envs
        num_sensors = len(self.sensor_names)
        device = env.device
        self.prev_forces = torch.zeros((num_envs, num_sensors, 3), device=device)
        self._has_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.prev_forces[:] = 0.0
            self._has_prev[:] = False
        else:
            self.prev_forces[env_ids, :, :] = 0.0
            self._has_prev[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_names: list[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        if sensor_names is None:
            sensor_names = self.sensor_names
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        if scale is None:
            scale = self.scale

        forces = contact_force_vectors(env, sensor_names)
        mag = torch.linalg.vector_norm(forces, dim=-1)
        in_range = (mag >= low) & (mag <= high)

        prev_mag = torch.linalg.vector_norm(self.prev_forces, dim=-1)
        diff = (mag - prev_mag) * in_range.to(mag.dtype)

        num_sensors = in_range.shape[1]
        all_in_range = in_range.all(dim=1)

        sum_sq = torch.sum(diff * diff, dim=1)
        rms = torch.where(all_in_range, torch.sqrt(sum_sq / float(num_sensors)), torch.zeros_like(sum_sq))

        per_env_score = (1.0 - torch.tanh(scale * rms)).clamp(min=0.0, max=1.0)
        per_env_score = torch.where(all_in_range & self._has_prev, per_env_score, torch.zeros_like(per_env_score))

        self.prev_forces = forces.clone()
        self._has_prev = torch.ones_like(self._has_prev)

        return per_env_score


class TanhConsecutiveActionConstancyReward(ManagerTermBase):
    """Stateful tanh reward that encourages consecutive actions to stay the same."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cast(dict[str, object], cfg.params)
        sensor_names = params.get("sensor_names")
        if not isinstance(sensor_names, list) or len(sensor_names) == 0:
            raise ValueError("'sensor_names' must be provided in params for TanhConsecutiveActionConstancyReward")
        action_name = params.get("action_name", "finger_targets")
        if not isinstance(action_name, str):
            raise ValueError("'action_name' must be a string in params for TanhConsecutiveActionConstancyReward")

        self.sensor_names = list(sensor_names)

        low = params.get("low", CONTACT_FORCE_LOW_THRESHOLD)
        self.low = float(low) if isinstance(low, (int, float, str)) else CONTACT_FORCE_LOW_THRESHOLD

        high = params.get("high", CONTACT_FORCE_HIGH_THRESHOLD)
        self.high = float(high) if isinstance(high, (int, float, str)) else CONTACT_FORCE_HIGH_THRESHOLD

        scale = params.get("scale", 20.0)
        self.scale = float(scale) if isinstance(scale, (int, float, str)) else 20.0

        self.action_name: str = action_name
        self.require_all_in_range: bool = bool(params.get("require_all_in_range", True))

        self._prev_actions: torch.Tensor | None = None
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if self._prev_actions is None:
            return
        if env_ids is None:
            self._prev_actions[:] = 0.0
            self._has_prev[:] = False
        else:
            self._prev_actions[env_ids, :] = 0.0
            self._has_prev[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_names: list[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        scale: float | None = None,
        action_name: str | None = None,
        require_all_in_range: bool | None = None,
    ) -> torch.Tensor:
        sensor_names = self.sensor_names if sensor_names is None else sensor_names
        low = self.low if low is None else low
        high = self.high if high is None else high
        scale = self.scale if scale is None else scale
        action_name = self.action_name if action_name is None else action_name
        require_all_in_range = self.require_all_in_range if require_all_in_range is None else require_all_in_range

        current_actions = env.action_manager.get_term(action_name).processed_actions
        num_envs, action_dim = current_actions.shape

        if (
            self._prev_actions is None
            or self._prev_actions.shape != current_actions.shape
            or self._prev_actions.device != current_actions.device
        ):
            self._prev_actions = torch.zeros((num_envs, action_dim), device=current_actions.device)
            self._has_prev = torch.zeros(num_envs, dtype=torch.bool, device=current_actions.device)

        mag = contact_force_magnitudes(env, sensor_names)
        in_range = (mag >= low) & (mag <= high)
        if require_all_in_range:
            forces_ok = in_range.all(dim=1)
        else:
            forces_ok = in_range.any(dim=1)

        delta = current_actions - self._prev_actions
        delta_rms = torch.sqrt(torch.mean(delta * delta, dim=1))
        reward = tanh_shaped_positive_reward(delta_rms, scale)
        reward = torch.where(forces_ok & self._has_prev, reward, torch.zeros_like(reward))

        self._prev_actions = current_actions.clone()
        self._has_prev = torch.ones_like(self._has_prev)
        return reward


def tanh_finger_symmetry_reward(env, asset_cfg: SceneEntityCfg, scale: float = 10.0) -> torch.Tensor:
    """Tanh reward encouraging symmetric finger joint positions (small spread)."""
    sym = finger_symmetry(env, asset_cfg)
    return tanh_shaped_positive_reward(sym, scale)


def tanh_action_proximity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    scale: float = 40.0,
    sensor_names: list[str] | None = None,
    touch_mask: bool = True,
) -> torch.Tensor:
    """Reward that encourages processed actions to be close to current joint positions."""
    fingers = env.scene[asset_cfg.name]
    joint_pos = fingers.data.joint_pos[:, asset_cfg.joint_ids]
    processed = env.action_manager.get_term("finger_targets").processed_actions

    per_joint_error = torch.abs(processed - joint_pos)
    per_joint_reward = tanh_shaped_positive_reward(per_joint_error, scale)

    if sensor_names is not None:
        mag = contact_force_magnitudes(env, sensor_names)
        touching = mag >= CONTACT_FORCE_LOW_THRESHOLD
        if touch_mask:
            mask = touching.to(per_joint_reward.dtype)
        else:
            mask = (~touching).to(per_joint_reward.dtype)
        per_joint_reward = per_joint_reward * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
    else:
        denom = torch.tensor(per_joint_reward.shape[1], device=per_joint_reward.device, dtype=per_joint_reward.dtype)

    return per_joint_reward.sum(dim=1) / denom


def tanh_contact_balance_reward(env, sensor_names: list[str], scale: float = 3.0) -> torch.Tensor:
    """Apply a tanh-shaped mapping to contact-force balance similarity."""
    balance = contact_force_balance(env, sensor_names)
    denom = float(torch.tanh(torch.tensor(scale)))
    return (torch.tanh(balance * scale) / denom).clamp(min=0.0, max=1.0)
