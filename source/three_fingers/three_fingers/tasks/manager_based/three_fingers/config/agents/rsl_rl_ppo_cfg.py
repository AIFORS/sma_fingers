# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDualRndOnPolicyRunnerCfg,
    RslRlDualRndPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlRndCfg,
)
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticRecurrentCfg


@configclass
class ThreeFingersGraspPPORunnerCfg(RslRlDualRndOnPolicyRunnerCfg):
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "rnd_object": ["rnd_object"], "rnd_robot": ["rnd_robot"]}
    max_iterations = 10000
    save_interval = 50
    experiment_name = "three_fingers_grasp"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        #rnn_type="lstm",
        #rnn_hidden_dim=512,
        #rnn_num_layers=1,
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlDualRndPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.98,
        desired_kl=0.05,
        max_grad_norm=1.0,
        rnd_object_cfg=RslRlRndCfg(
            weight=-25.0,
            #weight_schedule=RslRlRndCfg.StepWeightScheduleCfg(final_step=200 * 24, final_value=0.0),
            reward_normalization=True,
            state_normalization=True,
            learning_rate=1.0e-3,
            predictor_hidden_dims=[128, 128],
            target_hidden_dims=[128, 128],
        ),
        rnd_robot_cfg=RslRlRndCfg(
            weight=5.0,
            #weight_schedule=RslRlRndCfg.StepWeightScheduleCfg(final_step=200 * 24, final_value=0.0),
            reward_normalization=True,
            state_normalization=True,
            learning_rate=1.0e-3,
            predictor_hidden_dims=[128, 128],
            target_hidden_dims=[128, 128],
        ),
    )
