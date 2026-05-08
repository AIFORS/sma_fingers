# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


# Register Gym environment for direct (non-RL) control of the three-finger mechanism.
gym.register(
    id="Isaac-Three-Fingers-Direct-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.manipulation.three_fingers.three_fingers_env_cfg:ThreeFingersEnvCfg"
        ),
    },
)

gym.register(
    id="Isaac-Three-Fingers-Grasp-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.manipulation.three_fingers.three_fingers_grasp_env_cfg:ThreeFingersGraspEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ThreeFingersGraspPPORunnerCfg",
    },
)
