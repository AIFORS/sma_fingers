# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import torch

from isaaclab.actuators.actuator_pd import ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.utils import configclass, DelayBuffer
from collections.abc import Sequence


class AsymmetricController(ImplicitActuator):
    """Implicit actuator that tapers closing motion with a linear profile."""

    cfg: "AsymmetricActuatorCfg"

    def __init__(self, cfg: "AsymmetricActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate delay buffers based on cfg values (history length is cfg.max_delay)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        # apply delay to incoming setpoints
        if control_action.joint_positions is not None:
            # compute per-environment delay based on normalized error across joints
            error = control_action.joint_positions - joint_pos
            abs_error = torch.abs(error)
            # normalize using configured max range and clamp to [0, 1]
            normalized_err = torch.clamp(abs_error / float(self.cfg.max_range), min=0.0, max=1.0)
            # reduce across joints to a single value per environment (use max)
            per_env_norm = torch.max(normalized_err, dim=-1)[0]
            # map normalized error to a time-lag in steps between min_delay and max_delay
            span = int(self.cfg.max_delay) - int(self.cfg.min_delay)
            time_lags = (per_env_norm * float(span) + float(self.cfg.min_delay)).round().to(dtype=torch.int)
            # ensure shape and device
            time_lags = time_lags.to(device=self._device)
            # set the same computed time lag for all delay buffers
            self.positions_delay_buffer.set_time_lag(time_lags)
            self.velocities_delay_buffer.set_time_lag(time_lags)
            self.efforts_delay_buffer.set_time_lag(time_lags)
            control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        else:
            return super().compute(control_action, joint_pos, joint_vel)
        if control_action.joint_velocities is not None:
            control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        if control_action.joint_efforts is not None:
            control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        error = control_action.joint_positions - joint_pos
        abs_error = torch.abs(error)
        tapped_error = torch.sign(error) * torch.pow(abs_error, 0.7)

        tapered_target = torch.where(joint_vel < 0, joint_pos + tapped_error * float(self.cfg.close_taper), control_action.joint_positions)
        control_action.joint_positions = tapered_target
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class AsymmetricActuatorCfg(ImplicitActuatorCfg):
    """Configuration for the asymmetric controller that slows closing."""

    class_type: type = AsymmetricController
    close_taper: float = MISSING
    # maximum range used to normalize joint errors for dynamic delay mapping
    max_range: float = MISSING
    # minimum delay (in physics time-steps) for buffering incoming actuator commands
    min_delay: int = 0
    # maximum delay (in physics time-steps) for buffering incoming actuator commands
    max_delay: int = 50 # 50
