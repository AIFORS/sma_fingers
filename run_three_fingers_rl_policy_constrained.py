# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import csv
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

from three_fingers_utils import (
    SENSOR_BASELINES,
    SENSOR_SCALES,
    denormalize_0_1,
    find_finger_joint4_ids,
    make_recordable_manager_env_class,
)


# Keep these local so we don't import isaaclab_tasks before SimulationApp exists.
DEFAULT_CONTACT_FORCE_LOW_THRESHOLD = 1.75
DEFAULT_CONTACT_FORCE_HIGH_THRESHOLD = 2.1


parser = argparse.ArgumentParser(description="Run ThreeFingers grasp RL policy from a checkpoint.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("/home/vsivtsov/Documents/kinova_isaac/logs/rsl_rl/three_fingers_grasp/2026-03-09_23-50-51/model_6200.pt"),
    help="Path to RL checkpoint .pt file.",
)
parser.add_argument("--run_time_s", type=float, default=30.0, help="Rollout time in seconds.")
parser.add_argument(
    "--policy_tanh_output",
    action="store_true",
    default=False,
    help="Apply tanh to policy output after final linear layer.",
)
parser.add_argument(
    "--action_delta_max",
    type=float,
    default=0.5,
    help="Max allowed deviation from current joint4 positions before env.step.",
)
parser.add_argument("--video", action="store_true", default=True, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=70_000, help="Length of recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=100_000, help="Interval between video recordings in steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401


class ActorPolicy(torch.nn.Module):
    def __init__(self, layer_dims: list[int], apply_tanh: bool = False):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError(f"Expected at least two layer dims, got: {layer_dims}")

        layers: list[torch.nn.Module] = []
        for idx in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
            if idx < len(layer_dims) - 2:
                layers.append(torch.nn.ELU())

        self.net = torch.nn.Sequential(*layers)
        self.apply_tanh = apply_tanh

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.net(obs)
        return torch.tanh(action) if self.apply_tanh else action


def _unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("actor_state_dict", "model_state_dict", "state_dict", "policy", "actor", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError("Unsupported checkpoint format. Expected dict-like checkpoint or state_dict.")


def _extract_actor_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("module.actor.", "actor.", "net."):
        keys = [key for key in state_dict.keys() if key.startswith(prefix)]
        if keys:
            return {key[len(prefix) :]: state_dict[key] for key in keys}

    linear_keys = [key for key in state_dict.keys() if re.match(r"^\d+\.(weight|bias)$", key)]
    if linear_keys:
        return {key: state_dict[key] for key in linear_keys}

    raise KeyError(
        "Could not find actor weights in checkpoint. Expected keys prefixed by 'actor.', "
        "'module.actor.', or 'net.'."
    )


def _infer_actor_dims(actor_state: dict[str, torch.Tensor]) -> list[int]:
    layer_ids = sorted(
        int(match.group(1))
        for key, value in actor_state.items()
        if value.ndim == 2 and (match := re.match(r"^(\d+)\.weight$", key)) is not None
    )
    if not layer_ids:
        raise ValueError("Actor state does not include linear weights in '<idx>.weight' format.")

    dims: list[int] = []
    for layer_id in layer_ids:
        weight = actor_state[f"{layer_id}.weight"]
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        if not dims:
            dims.append(in_features)
        dims.append(out_features)
    return dims


def load_policy(path: Path, device: torch.device | str, apply_tanh: bool = False) -> tuple[ActorPolicy, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device)
    state_dict = _unwrap_state_dict(checkpoint)
    actor_state = _extract_actor_state(state_dict)
    layer_dims = _infer_actor_dims(actor_state)

    policy = ActorPolicy(layer_dims, apply_tanh=apply_tanh).to(device)
    remapped = {f"net.{key}": value for key, value in actor_state.items()}
    missing_keys, unexpected_keys = policy.load_state_dict(remapped, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load policy cleanly. "
            f"Missing keys: {missing_keys}; Unexpected keys: {unexpected_keys}."
        )

    policy.eval()
    return policy, layer_dims[0], layer_dims[-1]


def _get_policy_obs(obs: dict[str, torch.Tensor | dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    policy_obs = obs["policy"]
    if isinstance(policy_obs, dict):
        if len(policy_obs) != 1:
            raise ValueError(f"Expected exactly one policy observation tensor, got keys: {list(policy_obs.keys())}")
        policy_obs = next(iter(policy_obs.values()))
    return torch.as_tensor(policy_obs, device=device, dtype=torch.float32)


def _contact_force_norm(sensor) -> float:
    force_matrix = sensor.data.force_matrix_w[0]
    if force_matrix.ndim == 3:
        force_vectors = force_matrix[:, 0, :]
    elif force_matrix.ndim == 2:
        force_vectors = force_matrix
    else:
        force_vectors = force_matrix.reshape(-1, 3)
    net_force_vector = force_vectors.sum(dim=0)
    return float(torch.linalg.vector_norm(net_force_vector).item())


def _flag_is_true(flag: torch.Tensor | np.ndarray | bool) -> bool:
    if isinstance(flag, torch.Tensor):
        return bool(torch.any(flag).item())
    if isinstance(flag, np.ndarray):
        return bool(np.any(flag))
    return bool(flag)


def _format_bool(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


@dataclass(frozen=True)
class SuccessCriteria:
    sensor_names: list[str]
    low: float
    high: float
    lin_vel_thresh: float
    ang_vel_thresh: float
    joint_vel_thresh: float
    action_tol: float
    joint_symmetry_tol: float | None
    action_name: str
    action_constancy_steps: int
    action_constancy_tol: float
    require_sphere_z_above_radius: bool


def _read_success_criteria(env_cfg) -> SuccessCriteria:
    success_params = getattr(getattr(env_cfg.rewards, "success", None), "params", {})
    joint_symmetry_tol_value = success_params.get("joint_symmetry_tol", None)
    joint_symmetry_tol = None if joint_symmetry_tol_value is None else float(joint_symmetry_tol_value)
    return SuccessCriteria(
        sensor_names=list(success_params.get("sensor_names", ["finger1_contact", "finger2_contact", "finger3_contact"])),
        low=float(success_params.get("low", DEFAULT_CONTACT_FORCE_LOW_THRESHOLD)),
        high=float(success_params.get("high", DEFAULT_CONTACT_FORCE_HIGH_THRESHOLD)),
        lin_vel_thresh=float(success_params.get("lin_vel_thresh", 0.01)),
        ang_vel_thresh=float(success_params.get("ang_vel_thresh", 0.01)),
        joint_vel_thresh=float(success_params.get("joint_vel_thresh", 0.01)),
        action_tol=float(success_params.get("action_tol", 0.02)),
        joint_symmetry_tol=joint_symmetry_tol,
        action_name=str(success_params.get("action_name", "finger_targets")),
        action_constancy_steps=max(1, int(success_params.get("action_constancy_steps", 6))),
        action_constancy_tol=max(0.0, float(success_params.get("action_constancy_tol", 0.04))),
        require_sphere_z_above_radius=bool(success_params.get("require_sphere_z_above_radius", False)),
    )


def denormalize_action_0_1(values, baseline, encoder_scale):
    """Denormalize action values while preserving exact zero commands as zero."""
    array = np.asarray(values, dtype=np.float32)
    denormalized = array * encoder_scale + baseline
    denormalized = np.where(array == 0.0, 0.0, denormalized)
    return float(denormalized) if np.isscalar(values) else denormalized


def plot_rl_tracking(
    plot_path: Path,
    times: np.ndarray,
    refs_real: list[np.ndarray],
    sims_denorm: list[np.ndarray],
    refs_norm: list[np.ndarray],
    sims_norm: list[np.ndarray],
    contact_forces: list[np.ndarray],
    joint4_efforts: list[np.ndarray],
):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    num_cols = 4
    fig, axes = plt.subplots(3, num_cols, figsize=(18, 9), sharex=True, constrained_layout=True)
    fig.set_facecolor("white")

    column_titles = [
        "Joint position [counts]",
        "Normalized position",
        "Contact force [N]",
        "Joint4 effort [Nm]",
    ]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title)

    for idx in range(3):
        axes[idx, 0].plot(
            times,
            refs_real[idx],
            label="reference",
            linewidth=2.0,
            color="#000000",
        )
        axes[idx, 0].plot(
            times,
            sims_denorm[idx],
            label="simulation",
            linewidth=2.0,
            linestyle="--",
            color="#6128e6",
        )
        axes[idx, 0].set_ylabel("Finger state (sensor units)")
        vals = np.concatenate((refs_real[idx], sims_denorm[idx]))
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)
        ymin_tick, ymax_tick = np.floor(ymin / 150.0) * 150.0, np.ceil(ymax / 150.0) * 150.0
        if ymin_tick == ymax_tick:
            ymin_tick -= 150.0
            ymax_tick += 150.0
        axes[idx, 0].set_yticks(np.arange(ymin_tick, ymax_tick + 0.1, 150.0))

        axes[idx, 1].plot(times, refs_norm[idx], label="reference", linewidth=2.0, color="#000000")
        axes[idx, 1].plot(times, sims_norm[idx], label="simulation", linewidth=2.0, linestyle="--", color="#6128e6")
        axes[idx, 1].set_ylim(0.0, 1.0)
        axes[idx, 1].set_yticks(np.arange(0.0, 1.01, 0.05))

        sim_contact_force = contact_forces[idx] * 1000.0
        axes[idx, 2].plot(times, sim_contact_force, label="sim contact |F|", linewidth=2.0, color="#e65c28")
        vals_f = np.asarray(sim_contact_force, dtype=np.float32)
        ymin_f, ymax_f = np.nanmin(vals_f), np.nanmax(vals_f)
        if ymin_f == ymax_f:
            ymin_f -= 1.0
            ymax_f += 1.0
        span_f = ymax_f - ymin_f
        contact_ymin = ymin_f - 0.05 * span_f
        contact_ymax = ymax_f + 0.05 * span_f
        axes[idx, 2].set_ylim(contact_ymin, contact_ymax)
        contact_tick_min = np.floor(contact_ymin / 250.0) * 250.0
        contact_tick_max = np.ceil(contact_ymax / 250.0) * 250.0
        axes[idx, 2].set_yticks(np.arange(contact_tick_min, contact_tick_max + 0.1, 250.0))
        axes[idx, 2].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[idx, 2].set_ylabel("Contact force [N]")

        effort_values = joint4_efforts[idx] * 1000.0
        axes[idx, 3].plot(
            times,
            effort_values,
            label="applied effort (joint4)",
            linewidth=1.6,
            color="#d62728",
        )
        vals_e = np.asarray(effort_values, dtype=np.float32)
        ymin_e, ymax_e = np.nanmin(vals_e), np.nanmax(vals_e)
        if ymin_e == ymax_e:
            ymin_e -= 1.0
            ymax_e += 1.0
        span_e = ymax_e - ymin_e
        effort_ymin = ymin_e - 0.05 * span_e
        effort_ymax = ymax_e + 0.05 * span_e
        axes[idx, 3].set_ylim(effort_ymin, effort_ymax)
        effort_tick_min = np.floor(effort_ymin / 500.0) * 500.0
        effort_tick_max = np.ceil(effort_ymax / 500.0) * 500.0
        axes[idx, 3].set_yticks(np.arange(effort_tick_min, effort_tick_max + 0.1, 500.0))
        axes[idx, 3].axhline(0.0, color="#888888", linewidth=0.6, linestyle="--")
        axes[idx, 3].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[idx, 3].set_ylabel("J4 effort [Nm]")

        axes[idx, 0].legend(frameon=False, ncol=2)
        if idx == 0:
            for col_idx in range(1, num_cols):
                axes[idx, col_idx].legend(frameon=False, ncol=1)

    for col in range(num_cols):
        axes[2, col].set_xlabel("Time [s]")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")
            ax.grid(alpha=0.25)

    fig.savefig(str(plot_path), dpi=300)
    plt.close(fig)


def main():
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.three_fingers.three_fingers_grasp_env_cfg import (
        ThreeFingersGraspEnvCfg,
    )

    BaseRecordableManagerEnv = make_recordable_manager_env_class(ManagerBasedRLEnv)

    class RecordableManagerEnv(BaseRecordableManagerEnv):
        def __init__(self, cfg, render_mode: str | None = None, sphere_radius: float | None = None):
            if sphere_radius is None:
                raise ValueError("sphere_radius must be provided for three-fingers RL evaluation.")
            # Needed by mdp.sphere_size/sphere_diameter observation terms during manager initialization.
            self._three_fingers_sphere_size_cache = torch.full(
                (cfg.scene.num_envs, 1),
                float(sphere_radius),
                dtype=torch.float32,
                device=cfg.sim.device,
            )
            super().__init__(cfg, render_mode=render_mode)

    log_path = Path("outputs/three_fingers_rl_policy_constrained.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = ThreeFingersGraspEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    success_criteria = _read_success_criteria(env_cfg)
    target_sphere_radius = 0.03 / 2
    if not hasattr(env_cfg.scene.sphere.spawn, "radius"):
        raise TypeError("three_fingers_grasp_env expects sphere.spawn to expose a 'radius' attribute")
    setattr(env_cfg.scene.sphere.spawn, "radius", target_sphere_radius)
    env_cfg.scene.sphere.init_state.pos = (0.0, 0.0, target_sphere_radius)
    # Keep sphere size fixed during evaluation.
    setattr(env_cfg.events, "reset_sphere_scale", None)
    # Evaluation mode: disable all termination terms so the rollout never auto-resets.
    for term_name in vars(env_cfg.terminations):
        if not term_name.startswith("_"):
            setattr(env_cfg.terminations, term_name, None)

    render_mode = "rgb_array" if args_cli.video else None
    base_env = RecordableManagerEnv(env_cfg, render_mode=render_mode, sphere_radius=target_sphere_radius)
    action_dim = base_env.action_manager.total_action_dim

    robot = base_env.scene["fingers"]
    joint4_ids = find_finger_joint4_ids(robot)
    sphere = base_env.scene["sphere"]
    action_term = base_env.action_manager.get_term(success_criteria.action_name)
    contact_sensors = [base_env.scene[sensor_name] for sensor_name in success_criteria.sensor_names]
    if len(contact_sensors) != 3:
        raise ValueError(f"Expected 3 contact sensors, got {len(contact_sensors)}: {success_criteria.sensor_names}")

    env: gym.Env = base_env
    if args_cli.video:
        video_folder = Path("outputs/videos/three_fingers_rl_policy_constrained")
        video_folder.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    try:
        policy, policy_input_dim, policy_output_dim = load_policy(
            args_cli.checkpoint,
            base_env.device,
            apply_tanh=args_cli.policy_tanh_output,
        )

        obs, _ = env.reset()
        policy_obs = _get_policy_obs(obs, base_env.device)
        if policy_obs.ndim != 2:
            raise ValueError(f"Expected policy observation with shape [num_envs, obs_dim], got {tuple(policy_obs.shape)}")
        if int(policy_obs.shape[1]) != policy_input_dim:
            raise ValueError(
                f"Policy input dim mismatch. Checkpoint expects {policy_input_dim}, env provides {int(policy_obs.shape[1])}."
            )
        if policy_output_dim != action_dim:
            raise ValueError(f"Policy output dim mismatch. Checkpoint outputs {policy_output_dim}, env expects {action_dim}.")

        rollout_steps = int(40.0 / base_env.step_dt)
        warmup_steps = int(2.0 / base_env.step_dt)
        cooldown_start = int(18.0 / base_env.step_dt)

        times: list[float] = []
        sphere_pos_hist = [[], [], []]
        actions_hist = [[] for _ in range(action_dim)]
        policy_obs_hist = [[] for _ in range(policy_input_dim)]
        contact_force_norms = [[] for _ in range(3)]
        joint4_efforts = [[] for _ in range(3)]
        action_history = deque(maxlen=success_criteria.action_constancy_steps)

        action = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
        for step_idx in tqdm(range(rollout_steps)):
            if step_idx < warmup_steps or step_idx >= cooldown_start:
                action.zero_()
            else:
                with torch.inference_mode():
                    action.copy_(policy(policy_obs))
                clamped_dims = min(action_dim, len(joint4_ids))
                if args_cli.action_delta_max is not None and clamped_dims > 0:
                    delta_max = float(args_cli.action_delta_max)
                    joint4_pos = robot.data.joint_pos[:, joint4_ids[:clamped_dims]]
                    lower = joint4_pos - delta_max
                    upper = joint4_pos + delta_max
                    action[:, :clamped_dims] = torch.clamp(action[:, :clamped_dims], min=lower, max=upper)

            obs, _, terminated, truncated, _ = env.step(action)
            if _flag_is_true(terminated) or _flag_is_true(truncated):
                raise RuntimeError(
                    "Termination/truncation occurred during evaluation, but evaluation mode expects none."
                )
            policy_obs = _get_policy_obs(obs, base_env.device)

            obs_np = policy_obs[0].detach().cpu().numpy().astype(np.float32)
            action_np = action[0].detach().cpu().numpy().astype(np.float32)
            sphere_pos_w = sphere.data.root_pos_w[0]
            sphere_x = float(sphere_pos_w[0].item())
            sphere_y = float(sphere_pos_w[1].item())
            sphere_z = float(sphere_pos_w[2].item())

            print(f"[step {step_idx}] sphere position (x,y,z): ({sphere_x:.3f}, {sphere_y:.3f}, {sphere_z:.3f})")
            print(
                "[step "
                f"{step_idx}] simulation state (observation): "
                f"{np.array2string(obs_np, precision=3, separator=', ')}"
            )
            print(
                "[step "
                f"{step_idx}] action: "
                f"{np.array2string(action_np, precision=3, separator=', ')}"
            )

            sphere_lin_speed = float(torch.linalg.vector_norm(sphere.data.root_lin_vel_w[0]).item())
            sphere_ang_speed = float(torch.linalg.vector_norm(sphere.data.root_ang_vel_w[0]).item())
            contact_magnitudes = np.asarray([_contact_force_norm(sensor) for sensor in contact_sensors], dtype=np.float32)
            joint_vel_abs = torch.abs(robot.data.joint_vel[0, joint4_ids]).detach().cpu().numpy().astype(np.float32)
            processed_actions = action_term.processed_actions[0].detach().cpu().numpy().astype(np.float32)
            joint_pos = robot.data.joint_pos[0, joint4_ids].detach().cpu().numpy().astype(np.float32)
            action_diff = np.abs(processed_actions - joint_pos)

            contact_low = success_criteria.low
            contact_high = success_criteria.high
            contact_ok = bool(np.all((contact_magnitudes >= contact_low) & (contact_magnitudes <= contact_high)))

            action_history.append(processed_actions.copy())
            history_ready = len(action_history) == success_criteria.action_constancy_steps
            if history_ready:
                history_arr = np.asarray(action_history, dtype=np.float32)
                action_span = history_arr.max(axis=0) - history_arr.min(axis=0)
                action_constancy_ok = bool(np.all(action_span <= success_criteria.action_constancy_tol))
            else:
                action_span = np.full(action_dim, np.nan, dtype=np.float32)
                action_constancy_ok = False

            radius_cache = getattr(base_env, "_three_fingers_sphere_size_cache", None)
            if radius_cache is None:
                sphere_radius = float(getattr(sphere.cfg.spawn, "radius", 0.0))
            else:
                sphere_radius = float(radius_cache[0, 0].item())
            sphere_height_ok = (sphere_z > sphere_radius) if success_criteria.require_sphere_z_above_radius else True

            lin_vel_ok = sphere_lin_speed <= success_criteria.lin_vel_thresh
            ang_vel_ok = sphere_ang_speed <= success_criteria.ang_vel_thresh
            joint_vel_ok = bool(np.all(joint_vel_abs <= success_criteria.joint_vel_thresh))
            action_tol_ok = bool(np.all(action_diff <= success_criteria.action_tol))
            joint_symmetry_tol = success_criteria.joint_symmetry_tol
            if joint_symmetry_tol is None:
                action_symmetric_ok = True
                pairwise_diff = np.zeros((action_dim, action_dim), dtype=np.float32)
            else:
                pairwise_diff = np.abs(processed_actions[:, None] - processed_actions[None, :])
                action_symmetric_ok = bool(np.all(pairwise_diff <= float(joint_symmetry_tol)))

            grasp_success_mask_ok = (
                contact_ok
                and lin_vel_ok
                and ang_vel_ok
                and joint_vel_ok
                and action_tol_ok
                and action_symmetric_ok
                and sphere_height_ok
            )
            final_success = grasp_success_mask_ok and action_constancy_ok

            if joint_symmetry_tol is None:
                joint_symmetry_line = (
                    f"[step {step_idx}] success check joint_symmetry_tol: "
                    "joint_symmetry_tol=None in config -> condition skipped in _grasp_success_mask -> PASS"
                )
            else:
                joint_symmetry_line = (
                    f"[step {step_idx}] success check joint_symmetry_tol: "
                    f"pairwise_abs_diff(processed_action)={np.array2string(pairwise_diff, precision=5, separator=', ')} "
                    f"<= {float(joint_symmetry_tol):.6f} (all pairs) -> {_format_bool(action_symmetric_ok)}"
                )

            print(
                f"[step {step_idx}] success check contact_range(low/high): "
                f"contact_magnitudes={np.array2string(contact_magnitudes, precision=5, separator=', ')} in "
                f"[{contact_low:.6f}, {contact_high:.6f}] (all sensors) -> {_format_bool(contact_ok)}"
            )

            print(
                f"[step {step_idx}] success check lin_vel_thresh: "
                f"||sphere_lin_vel||={sphere_lin_speed:.6f} <= {success_criteria.lin_vel_thresh:.6f} "
                f"-> {_format_bool(lin_vel_ok)}"
            )
            print(
                f"[step {step_idx}] success check ang_vel_thresh: "
                f"||sphere_ang_vel||={sphere_ang_speed:.6f} <= {success_criteria.ang_vel_thresh:.6f} "
                f"-> {_format_bool(ang_vel_ok)}"
            )
            print(
                f"[step {step_idx}] success check joint_vel_thresh: "
                f"abs(joint_vel_joint4)={np.array2string(joint_vel_abs, precision=5, separator=', ')} "
                f"<= {success_criteria.joint_vel_thresh:.6f} (all dims) -> {_format_bool(joint_vel_ok)}"
            )
            print(
                f"[step {step_idx}] success check action_tol: "
                f"abs(processed_action - joint_pos)={np.array2string(action_diff, precision=5, separator=', ')} "
                f"<= {success_criteria.action_tol:.6f} (all dims) -> {_format_bool(action_tol_ok)}"
            )
            print(joint_symmetry_line)
            print(
                f"[step {step_idx}] success check action_constancy_steps: "
                f"history_fill={len(action_history)}/{success_criteria.action_constancy_steps} "
                f"(require full window) -> {_format_bool(history_ready)}"
            )
            print(
                f"[step {step_idx}] success check action_constancy_tol: "
                f"action_span_over_window={np.array2string(action_span, precision=5, separator=', ')} "
                f"<= {success_criteria.action_constancy_tol:.6f} (all dims, window ready={history_ready}) "
                f"-> {_format_bool(action_constancy_ok)}"
            )
            print(
                f"[step {step_idx}] success check require_sphere_z_above_radius: "
                f"required={success_criteria.require_sphere_z_above_radius}, "
                f"sphere_z={sphere_z:.6f}, sphere_radius={sphere_radius:.6f}, "
                f"comparison=(sphere_z > sphere_radius) -> {_format_bool(sphere_height_ok)}"
            )
            print(f"[step {step_idx}] _grasp_success_mask conditions overall: {_format_bool(grasp_success_mask_ok)}")
            print(
                f"[step {step_idx}] GraspSuccessWithActionConstancy overall (_grasp_success_mask AND action_constancy): "
                f"{_format_bool(final_success)}"
            )

            times.append((step_idx + 1) * base_env.step_dt)
            sphere_pos_hist[0].append(sphere_x)
            sphere_pos_hist[1].append(sphere_y)
            sphere_pos_hist[2].append(sphere_z)

            for dim_idx in range(action_dim):
                actions_hist[dim_idx].append(float(action_np[dim_idx]))
            for dim_idx in range(policy_input_dim):
                policy_obs_hist[dim_idx].append(float(obs_np[dim_idx]))
            for sensor_idx, sensor in enumerate(contact_sensors):
                contact_force_norms[sensor_idx].append(_contact_force_norm(sensor))
            for finger_idx, joint_id in enumerate(joint4_ids):
                joint4_efforts[finger_idx].append(float(robot.data.applied_torque[0, joint_id].item()))

        times_arr = np.asarray(times, dtype=np.float32)
        sphere_pos_arr = [np.asarray(values, dtype=np.float32) for values in sphere_pos_hist]
        actions_arr = [np.asarray(values, dtype=np.float32) for values in actions_hist]
        policy_obs_arr = [np.asarray(values, dtype=np.float32) for values in policy_obs_hist]
        contact_force_norms = [np.asarray(values, dtype=np.float32) for values in contact_force_norms]
        joint4_efforts = [np.asarray(values, dtype=np.float32) for values in joint4_efforts]

        with log_path.open("w", newline="") as file:
            writer = csv.writer(file)
            header = ["time_s", "sphere_x", "sphere_y", "sphere_z"]
            header.extend([f"action{idx + 1}_norm" for idx in range(action_dim)])
            header.extend([f"action{idx + 1}_denorm" for idx in range(action_dim)])
            header.extend([f"policy_obs_{idx + 1}" for idx in range(policy_input_dim)])
            header.extend([f"contact_finger{idx + 1}_f" for idx in range(3)])
            header.extend([f"joint4_effort_finger{idx + 1}" for idx in range(3)])
            writer.writerow(header)

            actions_denorm_all = [
                np.asarray(denormalize_action_0_1(values, SENSOR_BASELINES[idx], SENSOR_SCALES[idx]), dtype=np.float32)
                for idx, values in enumerate(actions_arr)
            ]
            for idx in range(times_arr.shape[0]):
                row = [
                    float(times_arr[idx]),
                    float(sphere_pos_arr[0][idx]),
                    float(sphere_pos_arr[1][idx]),
                    float(sphere_pos_arr[2][idx]),
                ]
                row.extend(float(actions_arr[act_idx][idx]) for act_idx in range(action_dim))
                row.extend(float(actions_denorm_all[act_idx][idx]) for act_idx in range(action_dim))
                row.extend(float(policy_obs_arr[obs_idx][idx]) for obs_idx in range(policy_input_dim))
                row.extend(float(contact_force_norms[sensor_idx][idx]) for sensor_idx in range(3))
                row.extend(float(joint4_efforts[finger_idx][idx]) for finger_idx in range(3))
                writer.writerow(row)

        if action_dim >= 3 and policy_input_dim >= 3:
            actions_first3 = [actions_arr[0], actions_arr[1], actions_arr[2]]
            obs_first3 = [policy_obs_arr[0], policy_obs_arr[1], policy_obs_arr[2]]
            actions_denorm = [
                np.asarray(denormalize_action_0_1(values, SENSOR_BASELINES[idx], SENSOR_SCALES[idx]), dtype=np.float32)
                for idx, values in enumerate(actions_first3)
            ]
            obs_denorm = [
                np.asarray(denormalize_0_1(values, SENSOR_BASELINES[idx], SENSOR_SCALES[idx]), dtype=np.float32)
                for idx, values in enumerate(obs_first3)
            ]

            plot_rl_tracking(
                Path("outputs/three_fingers_rl_policy_constrained_tracking.png"),
                times_arr,
                actions_denorm,
                obs_denorm,
                actions_first3,
                obs_first3,
                contact_force_norms,
                joint4_efforts,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
