import csv
import inspect
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

SENSOR_BASELINES = (2256.0, 2120.0, 2748.0) #(2210.0, 2080.0, 2665.0)
_SENSOR_FULL_SCALES = (2865.0, 2500.0, 2944.0)
# encoder-scale = full_scale - baseline for the hardcoded-baseline path
SENSOR_SCALES = (
    _SENSOR_FULL_SCALES[0] - SENSOR_BASELINES[0],
    _SENSOR_FULL_SCALES[1] - SENSOR_BASELINES[1],
    _SENSOR_FULL_SCALES[2] - SENSOR_BASELINES[2],
)


def normalize_0_1(values, baseline, encoder_scale, clamp_to_unit=False):
    array = np.asarray(values, dtype=np.float32)
    normalized = (array - baseline) / encoder_scale
    normalized = np.clip(normalized, 0.0, 1.0) if clamp_to_unit else normalized
    return float(normalized) if np.isscalar(values) else normalized


def denormalize_0_1(values, baseline, encoder_scale):
    array = np.asarray(values, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    denormalized = array * encoder_scale + baseline
    return float(denormalized) if np.isscalar(values) else denormalized


def parse_experiments(value: str) -> list[int]:
    return [int(chunk) for chunk in value.split(",") if chunk]


def load_reference_sequences(csv_path: Path, sampling: int, experiments: list[int], estimate_baselines: bool = False):
    data: Any = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32)
    times_full = data["time"]
    baseline_mask = (times_full >= 2.0) & (times_full <= 6.0)
    valid_mask = times_full >= 6.0
    times = times_full[valid_mask][::sampling].astype(np.float32)
    baselines, refs_real, refs_norm, real_data, real_norm, force_refs, force_real = [], [], [], [], [], [], []
    for idx, ref_col in enumerate(("ref1", "ref2", "ref3"), start=1):
        if estimate_baselines:
            baseline = float(data[f"finger{idx}_exp{experiments[0]}"][baseline_mask].mean())
            full_scale = _SENSOR_FULL_SCALES[idx - 1]
            encoder_scale = full_scale - baseline
        else:
            baseline = float(SENSOR_BASELINES[idx - 1])
            encoder_scale = SENSOR_SCALES[idx - 1]

        baselines.append(baseline)
        ref_values = data[ref_col][valid_mask][::sampling].astype(np.float32)
        refs_real.append(ref_values)
        refs_norm.append(normalize_0_1(ref_values, baseline, encoder_scale, clamp_to_unit=True))
        finger_real = {exp: data[f"finger{idx}_exp{exp}"][valid_mask][::sampling].astype(np.float32) for exp in experiments}
        finger_norm = {exp: normalize_0_1(values, baseline, encoder_scale) for exp, values in finger_real.items()}
        force_refs.append(data[f"force_ref{idx}"][valid_mask][::sampling].astype(np.float32))
        force_real.append({exp: data[f"force_finger{idx}_exp{exp}"][valid_mask][::sampling].astype(np.float32) for exp in experiments})
        real_data.append(finger_real)
        real_norm.append(finger_norm)
    return times, refs_real, refs_norm, real_data, real_norm, baselines, force_refs, force_real


def plot_tracking(
    plot_path: Path,
    times,
    refs_real,
    sims_denorm,
    real_data_list,
    refs_norm,
    sims_norm,
    real_norm_list,
    contact_forces,
    force_refs_list,
    force_real_list,
    joint4_efforts_list,
    experiments,
):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    num_cols = 4
    fig, axes = plt.subplots(3, num_cols, figsize=(18, 9), sharex=True, constrained_layout=True)
    fig.set_facecolor("white")

    cmap = cm.get_cmap("tab10")
    column_titles = [
        "Joint position [counts]",
        "Normalized position",
        "Contact force [N]",
        "Joint4 effort [Nm]",
    ]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title)

    for i in range(3):
        axes[i, 0].plot(times, refs_real[i], label="reference", linewidth=2.0, color="#000000")
        axes[i, 0].plot(times, sims_denorm[i], label="simulation", linewidth=2.0, linestyle="--", color="#6128e6")
        for idx, exp in enumerate(experiments):
            axes[i, 0].plot(times, real_data_list[i][exp], label=f"real exp{exp}", linewidth=1.2, color=cmap(idx % 10), alpha=0.3)
        axes[i, 0].set_ylabel("Finger state (sensor units)")
        vals = np.concatenate((refs_real[i], sims_denorm[i], *[real_data_list[i][exp] for exp in experiments]))
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)
        ymin_tick, ymax_tick = np.floor(ymin / 150.0) * 150.0, np.ceil(ymax / 150.0) * 150.0
        if ymin_tick == ymax_tick:
            ymin_tick -= 150.0
            ymax_tick += 150.0
        axes[i, 0].set_yticks(np.arange(ymin_tick, ymax_tick + 0.1, 150.0))

        axes[i, 1].plot(times, refs_norm[i], label="reference", linewidth=2.0, color="#000000")
        axes[i, 1].plot(times, sims_norm[i], label="simulation", linewidth=2.0, linestyle="--", color="#6128e6")
        for idx, exp in enumerate(experiments):
            axes[i, 1].plot(times, real_norm_list[i][exp], label=f"real exp{exp} (norm)", linewidth=1.2, color=cmap(idx % 10), alpha=0.3)
        axes[i, 1].set_ylim(0.0, 1.0)
        axes[i, 1].set_yticks(np.arange(0.0, 1.01, 0.05))

        sim_contact_force = contact_forces[i] * 1000.0
        axes[i, 2].plot(times, sim_contact_force, label="sim contact |F|", linewidth=2.0, color="#e65c28")
        axes[i, 2].plot(times, force_refs_list[i], label="ref force", linewidth=2.0, color="#000000")
        for idx, exp in enumerate(experiments):
            axes[i, 2].plot(times, force_real_list[i][exp], label=f"real force exp{exp}", linewidth=1.2, color=cmap(idx % 10), alpha=0.3)
        vals_f = np.concatenate((sim_contact_force, force_refs_list[i], *[force_real_list[i][exp] for exp in experiments]))
        ymin_f, ymax_f = np.nanmin(vals_f), np.nanmax(vals_f)
        if ymin_f == ymax_f:
            ymin_f -= 1.0
            ymax_f += 1.0
        span_f = ymax_f - ymin_f
        contact_ymin = ymin_f - 0.05 * span_f
        contact_ymax = ymax_f + 0.05 * span_f
        axes[i, 2].set_ylim(contact_ymin, contact_ymax)
        contact_tick_min = np.floor(contact_ymin / 250.0) * 250.0
        contact_tick_max = np.ceil(contact_ymax / 250.0) * 250.0
        axes[i, 2].set_yticks(np.arange(contact_tick_min, contact_tick_max + 0.1, 250.0))
        axes[i, 2].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[i, 2].set_ylabel("Contact force [N]")

        effort_values = joint4_efforts_list[i] * 1000.0
        axes[i, 3].plot(
            times,
            effort_values,
            label="applied effort (joint4)",
            linewidth=1.6,
            color="#d62728",
        )
        axes[i, 3].plot(times, force_refs_list[i], label="ref force", linewidth=2.0, color="#000000")
        for idx, exp in enumerate(experiments):
            axes[i, 3].plot(
                times,
                force_real_list[i][exp],
                label=f"real force exp{exp}",
                linewidth=1.2,
                color=cmap(idx % 10),
                alpha=0.3,
            )
        vals_e = np.concatenate((effort_values, force_refs_list[i], *[force_real_list[i][exp] for exp in experiments]))
        ymin_e, ymax_e = np.nanmin(vals_e), np.nanmax(vals_e)
        if ymin_e == ymax_e:
            ymin_e -= 1.0
            ymax_e += 1.0
        span_e = ymax_e - ymin_e
        effort_ymin = ymin_e - 0.05 * span_e
        effort_ymax = ymax_e + 0.05 * span_e
        axes[i, 3].set_ylim(effort_ymin, effort_ymax)
        effort_tick_min = np.floor(effort_ymin / 500.0) * 500.0
        effort_tick_max = np.ceil(effort_ymax / 500.0) * 500.0
        axes[i, 3].set_yticks(np.arange(effort_tick_min, effort_tick_max + 0.1, 500.0))
        axes[i, 3].axhline(0.0, color="#888888", linewidth=0.6, linestyle="--")
        axes[i, 3].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[i, 3].set_ylabel("J4 effort [Nm]")

        axes[i, 0].legend(frameon=False, ncol=2)
        if i == 0:
            for col_idx in range(1, num_cols):
                axes[i, col_idx].legend(frameon=False, ncol=1)

    for col in range(num_cols):
        axes[2, col].set_xlabel("Time [s]")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")
            ax.grid(alpha=0.25)
    fig.savefig(str(plot_path), dpi=300)
    plt.close(fig)


def make_recordable_manager_env_class(manager_based_env_cls):
    init_signature = inspect.signature(manager_based_env_cls.__init__)
    supports_render_mode = "render_mode" in init_signature.parameters

    class RecordableManagerEnv(manager_based_env_cls, gym.Env):
        metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 0.0}

        def __init__(self, cfg, render_mode: str | None = None):
            self.render_mode = render_mode
            if supports_render_mode:
                super().__init__(cfg, render_mode=render_mode)
            else:
                super().__init__(cfg)
            self.metadata["render_fps"] = 1.0 / self.step_dt

        def render(self, recompute: bool = False):
            if not self.sim.has_rtx_sensors() and not recompute:
                self.sim.render()
            if self.render_mode is None or self.render_mode == "human":
                return None
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is '{self.sim.render_mode.name}'."
                    f" Please set the simulation render mode to: '{self.sim.RenderMode.PARTIAL_RENDERING.name}' or"
                    f" '{self.sim.RenderMode.FULL_RENDERING.name}'. If running headless, make sure --enable_cameras is set."
                )
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach(cast(Any, self._render_product))
            rgb_data = self._rgb_annotator.get_data()
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            return (
                np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
                if rgb_data.size == 0
                else rgb_data[:, :, :3]
            )

    return RecordableManagerEnv


class GymStepAdapter(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 0.0}

    def __init__(self, inner_env, action_dim: int):
        self.inner_env = inner_env
        self.render_mode = inner_env.render_mode
        self.metadata["render_fps"] = 1.0 / inner_env.step_dt
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
        obs_shape = (inner_env.num_envs, 3)
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)}
        )

    def reset(self, *, seed=None, options=None):  # noqa: D401
        obs, extras = self.inner_env.reset()
        return obs, extras

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, device=self.inner_env.device)
        obs, extras = self.inner_env.step(action)
        return obs, 0.0, False, False, extras

    def render(self):
        return self.inner_env.render()

    def close(self):
        self.inner_env.close()


def find_finger_joint_ids(robot):
    finger_joint_ids = []
    for finger_idx in range(1, 4):
        joint_ids, _ = robot.find_joints(
            [
                f"finger{finger_idx}_joint1",
                f"finger{finger_idx}_joint2",
                f"finger{finger_idx}_joint3",
                f"finger{finger_idx}_joint4",
            ]
        )
        finger_joint_ids.append(joint_ids)
    return finger_joint_ids


def find_finger_joint4_ids(robot):
    joint4_ids = []
    for finger_idx in range(1, 4):
        joint_ids, _ = robot.find_joints([f"finger{finger_idx}_joint4"])
        joint4_ids.append(int(joint_ids[0]))
    return joint4_ids


def write_tracking_csv(
    log_path: Path,
    times,
    refs_real,
    refs_norm,
    sim_denorm,
    simulated_states,
    contact_force_norms,
    force_refs,
    real_data,
    real_norm,
    force_real,
    joint4_efforts,
    experiments,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "time_s",
            "ref1",
            "ref2",
            "ref3",
            "ref1_norm",
            "ref2_norm",
            "ref3_norm",
            "sim_finger1",
            "sim_finger2",
            "sim_finger3",
            "sim_finger1_norm",
            "sim_finger2_norm",
            "sim_finger3_norm",
            "contact_finger1_f",
            "contact_finger2_f",
            "contact_finger3_f",
            "joint4_effort_finger1",
            "joint4_effort_finger2",
            "joint4_effort_finger3",
            "force_ref1",
            "force_ref2",
            "force_ref3",
        ]
        for finger_idx in (1, 2, 3):
            for exp in experiments:
                header.append(f"finger{finger_idx}_exp{exp}")
                header.append(f"finger{finger_idx}_exp{exp}_norm")
        for finger_idx in (1, 2, 3):
            for exp in experiments:
                header.append(f"force_finger{finger_idx}_exp{exp}")
        writer.writerow(header)
        for idx in range(times.shape[0]):
            row = [
                float(times[idx]),
                float(refs_real[0][idx]),
                float(refs_real[1][idx]),
                float(refs_real[2][idx]),
                float(refs_norm[0][idx]),
                float(refs_norm[1][idx]),
                float(refs_norm[2][idx]),
                float(sim_denorm[0][idx]),
                float(sim_denorm[1][idx]),
                float(sim_denorm[2][idx]),
                float(simulated_states[0][idx]),
                float(simulated_states[1][idx]),
                float(simulated_states[2][idx]),
                float(contact_force_norms[0][idx]),
                float(contact_force_norms[1][idx]),
                float(contact_force_norms[2][idx]),
                float(joint4_efforts[0][idx]),
                float(joint4_efforts[1][idx]),
                float(joint4_efforts[2][idx]),
                float(force_refs[0][idx]),
                float(force_refs[1][idx]),
                float(force_refs[2][idx]),
            ]
            for finger_idx in (0, 1, 2):
                for exp in experiments:
                    row.append(float(real_data[finger_idx][exp][idx]))
                    row.append(float(real_norm[finger_idx][exp][idx]))
            for finger_idx in (0, 1, 2):
                for exp in experiments:
                    row.append(float(force_real[finger_idx][exp][idx]))
            writer.writerow(row)