# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

from three_fingers_utils import (
    SENSOR_BASELINES,
    SENSOR_SCALES,
    denormalize_0_1,
    find_finger_joint4_ids,
    GymStepAdapter,
    make_recordable_manager_env_class,
    parse_experiments,
    plot_tracking,
    write_tracking_csv,
)


def unit_interval_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"Expected value in [0, 1], got {value!r}.")
    return parsed


parser = argparse.ArgumentParser(
    description=(
        "Run ThreeFingers environment with a synthetic reference profile: "
        "3 seconds of zero action followed by 7 seconds of fixed per-finger actions."
    )
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--plot_experiments", type=str, default="1,2,3,4,5", help="Comma-separated experiment indices to plot (e.g., 1,3,5).")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during the rollout.")
parser.add_argument("--video_length", type=int, default=70_000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100_000, help="Interval between video recordings (in steps).")
parser.add_argument("--finger1_action", type=unit_interval_value, default=0.9, help="Normalized command for finger 1 in [0, 1].")
parser.add_argument("--finger2_action", type=unit_interval_value, default=0.9, help="Normalized command for finger 2 in [0, 1].")
parser.add_argument("--finger3_action", type=unit_interval_value, default=0.9, help="Normalized command for finger 3 in [0, 1].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401


def main():
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab_tasks.manager_based.manipulation.three_fingers.three_fingers_env_cfg import ThreeFingersEnvCfg

    RecordableManagerEnv = make_recordable_manager_env_class(ManagerBasedEnv)

    experiments = parse_experiments(args_cli.plot_experiments)
    if len(experiments) == 0:
        raise ValueError("--plot_experiments must provide at least one experiment index.")
    selected_exp = experiments[0]

    commanded_actions = np.asarray(
        [args_cli.finger1_action, args_cli.finger2_action, args_cli.finger3_action],
        dtype=np.float32,
    )

    log_path = Path("outputs/three_figners_small_sphere_arg_action_tracking.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = ThreeFingersEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Keep this script deterministic and aligned with the 1.0-style controlled setup.
    target_sphere_radius = 0.03 / 2
    if not hasattr(env_cfg.scene.sphere.spawn, "radius"):
        raise TypeError("three_fingers_env expects sphere.spawn to expose a 'radius' attribute")
    env_cfg.scene.sphere.spawn.radius = target_sphere_radius
    env_cfg.scene.sphere.init_state.pos = (0.0, 0.0, target_sphere_radius)

    render_mode = "rgb_array" if args_cli.video else None
    base_env = RecordableManagerEnv(env_cfg, render_mode=render_mode)
    action_dim = len(env_cfg.actions.finger_targets.joint_names)

    step_dt = float(base_env.step_dt)
    zero_steps = int(np.round(3.0 / step_dt))
    active_steps = int(np.round(7.0 / step_dt))
    release_steps = int(np.round(20.0 / step_dt))
    total_steps = zero_steps + active_steps + release_steps
    times = np.arange(total_steps, dtype=np.float32) * np.float32(step_dt)

    refs_norm = []
    refs_real = []
    for finger_idx in range(3):
        ref_profile = np.zeros(total_steps, dtype=np.float32)
        ref_profile[zero_steps:] = commanded_actions[finger_idx]
        ref_profile[zero_steps + active_steps :] = 0.0
        refs_norm.append(ref_profile)
        refs_real.append(
            np.asarray(
                denormalize_0_1(ref_profile, SENSOR_BASELINES[finger_idx], SENSOR_SCALES[finger_idx]),
                dtype=np.float32,
            )
        )

    real_data = []
    real_norm = []
    force_refs = []
    force_real = []
    for finger_idx in range(3):
        finger_real = {exp: refs_real[finger_idx].copy() for exp in experiments}
        finger_norm = {exp: refs_norm[finger_idx].copy() for exp in experiments}
        real_data.append(finger_real)
        real_norm.append(finger_norm)

        force_profile = np.zeros(total_steps, dtype=np.float32)
        force_refs.append(force_profile)
        force_real.append({exp: force_profile.copy() for exp in experiments})

    robot = base_env.scene["fingers"]
    sphere = base_env.scene["sphere"]
    contact_sensors = [
        base_env.scene["finger1_contact"],
        base_env.scene["finger2_contact"],
        base_env.scene["finger3_contact"],
    ]
    joint4_ids = find_finger_joint4_ids(robot)
    joint4_efforts = [[] for _ in range(3)]

    env = GymStepAdapter(base_env, action_dim)
    if args_cli.video:
        video_folder = Path("outputs/videos/three_figners_small_sphere_arg_action")
        video_folder.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    try:
        obs, _ = env.reset()
        action = torch.empty((base_env.num_envs, action_dim), device=base_env.device)
        simulated_states = [[] for _ in range(3)]
        contact_force_norms = [[] for _ in range(3)]

        for step_idx in tqdm(range(total_steps)):
            action.fill_(0.0)
            action[:, 0].fill_(float(refs_norm[0][step_idx]))
            action[:, 1].fill_(float(refs_norm[1][step_idx]))
            action[:, 2].fill_(float(refs_norm[2][step_idx]))
            obs, _, _, _, _ = env.step(action)

            # Record states from observation.
            joint_obs = torch.as_tensor(obs["policy"], device="cpu")[0]
            f1 = float(joint_obs[0].item())
            f2 = float(joint_obs[1].item())
            f3 = float(joint_obs[2].item())
            simulated_states[0].append(f1)
            simulated_states[1].append(f2)
            simulated_states[2].append(f3)

            sphere_pos_w = sphere.data.root_pos_w[0]
            sphere_x = float(sphere_pos_w[0].item())
            sphere_y = float(sphere_pos_w[1].item())
            sphere_z = float(sphere_pos_w[2].item())
            real_f1 = float(real_norm[0][selected_exp][step_idx])
            real_f2 = float(real_norm[1][selected_exp][step_idx])
            real_f3 = float(real_norm[2][selected_exp][step_idx])
            action_values = (
                float(action[0, 0].item()),
                float(action[0, 1].item()),
                float(action[0, 2].item()),
            )
            print(f"[step {step_idx}] sphere position (x,y,z): ({sphere_x:.3f}, {sphere_y:.3f}, {sphere_z:.3f})")
            print(f"[step {step_idx}] simulation state (normalized): ({f1:.3f}, {f2:.3f}, {f3:.3f})")
            print(f"[step {step_idx}] real normalized state: ({real_f1:.3f}, {real_f2:.3f}, {real_f3:.3f})")
            print(
                f"[step {step_idx}] action (normalized): "
                f"({action_values[0]:.3f}, {action_values[1]:.3f}, {action_values[2]:.3f})"
            )

            for sensor_idx, sensor in enumerate(contact_sensors):
                force_matrix = sensor.data.force_matrix_w[0]
                if force_matrix.ndim == 3:
                    force_vectors = force_matrix[:, 0, :]
                elif force_matrix.ndim == 2:
                    force_vectors = force_matrix
                else:
                    force_vectors = force_matrix.reshape(-1, 3)
                net_force_vector = force_vectors.sum(dim=0)
                contact_force_norms[sensor_idx].append(float(torch.linalg.vector_norm(net_force_vector).item()))

            for finger_idx, joint_id in enumerate(joint4_ids):
                joint4_efforts[finger_idx].append(float(robot.data.applied_torque[0, joint_id].item()))

        simulated_states = [np.asarray(values, dtype=np.float32) for values in simulated_states]
        contact_force_norms = [np.asarray(values, dtype=np.float32) for values in contact_force_norms]
        sim_denorm = []
        for idx, values in enumerate(simulated_states):
            sim_denorm.append(np.asarray(denormalize_0_1(values, SENSOR_BASELINES[idx], SENSOR_SCALES[idx]), dtype=np.float32))
        joint4_efforts = [np.asarray(values, dtype=np.float32) for values in joint4_efforts]

        write_tracking_csv(
            log_path,
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
        )

        # Single consolidated figure: 3 rows (fingers) x 4 columns
        # (sensor, normalized, contact force, joint4 effort).
        plot_tracking(
            Path("outputs/three_figners_small_sphere_arg_action_tracking.png"),
            times,
            refs_real,
            sim_denorm,
            real_data,
            refs_norm,
            simulated_states,
            real_norm,
            contact_force_norms,
            force_refs,
            force_real,
            joint4_efforts,
            experiments,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
