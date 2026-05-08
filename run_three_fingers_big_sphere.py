# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import gymnasium as gym
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from three_fingers_utils import (
    SENSOR_BASELINES,
    SENSOR_SCALES,
    denormalize_0_1,
    find_finger_joint4_ids,
    GymStepAdapter,
    load_reference_sequences,
    make_recordable_manager_env_class,
    parse_experiments,
    plot_tracking,
    write_tracking_csv,
)

parser = argparse.ArgumentParser(description="Run ThreeFingers environment with multi-finger reference tracking from Dec25_big.csv.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sampling_time", type=int, default=10, help="Sampling interval for CSV data (load every N-th element).")
parser.add_argument("--plot_experiments", type=str, default="1,2,3,4,5", help="Comma-separated experiment indices to plot (e.g., 1,3,5).")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during the rollout.")
parser.add_argument("--video_length", type=int, default=70_000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100_000, help="Interval between video recordings (in steps).")
parser.add_argument("--estimate_baselines", action="store_true", default=False, help="Estimate sensor baselines from CSV (legacy behavior).")
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
    from three_fingers.tasks.manager_based.three_fingers import ThreeFingersEnvCfg
    RecordableManagerEnv = make_recordable_manager_env_class(ManagerBasedEnv)

    experiments = parse_experiments(args_cli.plot_experiments)
    selected_exp = experiments[0]
    times, refs_real, refs_norm, real_data, real_norm, baselines, force_refs, force_real = load_reference_sequences(
        Path("finger_open_close/Dec25_big.csv"), args_cli.sampling_time, experiments
    )

    log_path = Path("outputs/three_figners_big_sphere_tracking.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = ThreeFingersEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Keep this script deterministic and aligned with the 1.0-style controlled setup.
    target_sphere_radius = 0.065 / 2
    if not hasattr(env_cfg.scene.sphere.spawn, "radius"):
        raise TypeError("three_fingers_env expects sphere.spawn to expose a 'radius' attribute")
    env_cfg.scene.sphere.spawn.radius = target_sphere_radius
    env_cfg.scene.sphere.init_state.pos = (0.0, 0.0, target_sphere_radius)

    render_mode = "rgb_array" if args_cli.video else None
    base_env = RecordableManagerEnv(env_cfg, render_mode=render_mode)
    action_dim = len(env_cfg.actions.finger_targets.joint_names)

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
        video_folder = Path("outputs/videos/three_figners_big_sphere")
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
        for step_idx in tqdm(range(refs_norm[0].shape[0])):
            action.fill_(0.0)
            action[:, 0].fill_(float(refs_norm[0][step_idx]))
            action[:, 1].fill_(float(refs_norm[1][step_idx]))
            action[:, 2].fill_(float(refs_norm[2][step_idx]))
            obs, _, _, _, _ = env.step(action)
            # record states from observation
            joint_obs = torch.as_tensor(obs["policy"], device="cpu")[0]
            f1, f2, f3 = float(joint_obs[0].item()), float(joint_obs[1].item()), float(joint_obs[2].item())
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
            if args_cli.estimate_baselines:
                full_scale = SENSOR_SCALES[idx] + SENSOR_BASELINES[idx]
                encoder_scale = full_scale - baselines[idx]
            else:
                encoder_scale = SENSOR_SCALES[idx]
            sim_denorm.append(np.asarray(denormalize_0_1(values, baselines[idx], encoder_scale), dtype=np.float32))
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

        # --- Calculate and print R^2 alignment ---
        print("\n" + "="*50)
        print("SIMULATION VS REAL WORLD ALIGNMENT (R^2 SCORE)")
        print("="*50)
        for finger_idx in range(3):
            # Fetch the real denormalized data for the tracked experiment sequence
            y_true = np.array(real_data[finger_idx][selected_exp])
            y_pred = sim_denorm[finger_idx]
            
            # Slice to the shortest sequence just in case to prevent shape mismatches
            min_len = min(len(y_true), len(y_pred))
            y_true_sliced = y_true[:min_len]
            y_pred_sliced = y_pred[:min_len]

            # Calculate R^2 using numpy (1 - SS_res / SS_tot)
            ss_res = np.sum((y_true_sliced - y_pred_sliced) ** 2)
            ss_tot = np.sum((y_true_sliced - np.mean(y_true_sliced)) ** 2)
            
            if ss_tot != 0:
                r2 = 1 - (ss_res / ss_tot)
            else:
                r2 = float('nan') # Handle edge-case where variance is 0
                
            print(f"Finger {finger_idx + 1} R^2: {r2:.4f}")
        print("="*50 + "\n")
        # ---------------------------------------------------


        # single consolidated figure: 3 rows (fingers) × 4 columns (sensor, normalized, contact force, joint4 effort)
        plot_tracking(
            Path("outputs/three_figners_big_sphere_tracking.png"),
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
        
        
        
        # =========================================================================
        # --- PUBLICATION READY PLOT ---
        # =========================================================================
        
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.5,
            'grid.alpha': 0.4,
            'grid.linestyle': '--'
        })
        
        fig = plt.figure(figsize=(4, 6))
        outer_gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)
        cmap = plt.get_cmap('tab10')
        
        share_x_ax = None
        axes_bot = []
        
        for i in range(3):
            # Increased hspace to 0.15 to prevent 100 and 2000 from intersecting
            # Changed height_ratios to [4, 1] so the bottom part is 4 times smaller
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[i], hspace=0.2, height_ratios=[4, 1]
            )
            
            if share_x_ax is None:
                ax_top = fig.add_subplot(inner_gs[0])
                ax_bot = fig.add_subplot(inner_gs[1], sharex=ax_top)
                share_x_ax = ax_bot
            else:
                ax_top = fig.add_subplot(inner_gs[0], sharex=share_x_ax)
                ax_bot = fig.add_subplot(inner_gs[1], sharex=share_x_ax)
                
            axes_bot.append(ax_bot)
            
            y_ref = refs_real[i]
            x_ref = times[:len(y_ref)]
            y_sim = sim_denorm[i]
            x_sim = times[:len(y_sim)]
            
            # --- UPPER PART: ALL SIGNALS ---
            for j, exp in enumerate(experiments):
                y_real = real_data[i][exp]
                x_real = times[:len(y_real)]
                ax_top.plot(x_real, y_real, color=cmap(j % 10), alpha=0.5, label=f"real {exp}")
            
            ax_top.plot(x_sim, y_sim, color="#947a23", linewidth=2.0, label="Simulation")
            ax_top.plot(x_ref, y_ref, color='black', linestyle='--', linewidth=1.5, label="Reference")
            
            # --- LOWER PART: ONLY REFERENCE ---
            ax_bot.plot(x_ref, y_ref, color='black', linestyle='--', linewidth=1.5, label="Reference")
            
            # Formatting for both axes
            for ax in (ax_top, ax_bot):
                ax.grid(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            # --- Static Cut Logic ---
            all_state_y = []
            for exp in experiments:
                all_state_y.extend(real_data[i][exp])
            all_state_y.extend(sim_denorm[i])
            
            s_max = max(all_state_y)
            s_pad = max((s_max - 2000) * 0.1, 10.0)
            
            # Top axis strictly cuts off at 2000
            ax_top.set_ylim(2000, s_max + s_pad)
            
            # Bottom axis strictly bounds from 0 to 100 to show the rising front
            ax_bot.set_ylim(0, 100)
            
            ax_top.spines['bottom'].set_visible(False)
            ax_bot.spines['top'].set_visible(False)
            
            ax_top.tick_params(labelbottom=False, bottom=False)
            if i < 2:  
                ax_bot.tick_params(labelbottom=False)
                
            # Add diagonal cut marks
            # Scale the y-offset for the bottom axis by the height ratio (4) 
            # so the diagonal line slopes remain visually parallel in pixel space.
            d_x = 0.015
            d_y_top = 0.015
            d_y_bot = d_y_top * 4 
            
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.0)
            ax_top.plot((-d_x, +d_x), (-d_y_top, +d_y_top), **kwargs)         
            kwargs.update(transform=ax_bot.transAxes)
            ax_bot.plot((-d_x, +d_x), (1 - d_y_bot, 1 + d_y_bot), **kwargs)   
            
            ax_top.set_ylabel(f"Finger {i+1} state\n(sensor units)", y=0, verticalalignment='center', labelpad=20)
            
        axes_bot[-1].set_xlabel("Time (s)")
        
        # Legend
        handles, labels = ax_top.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), 
                   loc='lower left', bbox_to_anchor=(0.95, 0.1), 
                   ncol=1, frameon=False)
        
        pub_path = Path("outputs/three_fingers_publication_plot.png")
        fig.savefig(pub_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # =========================================================================

        
    except Exception as exc:  # noqa: BLE001
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
