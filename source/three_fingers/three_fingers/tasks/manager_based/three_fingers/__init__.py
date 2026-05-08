# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the 3-fingers manipulation environment."""

from .single_finger_env_cfg import SingleFingerEnvCfg
from .three_fingers_env_cfg import ThreeFingersEnvCfg

__all__ = ["SingleFingerEnvCfg", "ThreeFingersEnvCfg"]

# Intentionally left minimal; configs are exposed through the package modules.
