# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from math import pi

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp


ACTIVE_JOINT_LOWER_DEG = -4.0
ACTIVE_JOINT_UPPER_DEG = 30.0
ACTIVE_JOINT_LOWER_RAD = ACTIVE_JOINT_LOWER_DEG / 180.0 * pi
ACTIVE_JOINT_RANGE_RAD = (ACTIVE_JOINT_UPPER_DEG - ACTIVE_JOINT_LOWER_DEG) / 180.0 * pi


@configclass
class ThreeFingersSceneCfg(InteractiveSceneCfg):
    """Scene with a three-finger mechanism."""

    fingers: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.065,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.5,
                angular_damping=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                min_torsional_patch_radius=0.05,
            ),
            visual_material=None,
            visual_material_path="{ENV_REGEX_NS}/Robot/fingers_on_base/Looks/sphere",
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.065)),
    )

    finger1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingers_on_base/finger_1/link5",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )

    finger2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingers_on_base/finger_2/link5",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )

    finger3_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fingers_on_base/finger_3/link5",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )


@configclass
class ThreeFingersActionsCfg:
    finger_targets: mdp.JointPositionActionCfg = MISSING  # type: ignore[assignment]


@configclass
class ThreeFingersObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        finger_state = ObsTerm(
            func=mdp.finger_state_normalized_from_open,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    joint_names=[
                        "finger1_joint4",
                        "finger2_joint4",
                        "finger3_joint4",
                    ],
                ),
                "lower_rad": ACTIVE_JOINT_LOWER_RAD,
                "upper_rad": ACTIVE_JOINT_UPPER_DEG / 180.0 * pi,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class ThreeFingersEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the three-finger joint-target environment."""

    scene: ThreeFingersSceneCfg = ThreeFingersSceneCfg(num_envs=1, env_spacing=1.5)
    observations: ThreeFingersObservationsCfg = ThreeFingersObservationsCfg()
    actions: ThreeFingersActionsCfg = ThreeFingersActionsCfg()

    def __post_init__(self):
        sim_frequency = 100
        self.sim.dt = 1 / sim_frequency

        control_freq = 10
        self.decimation = int(sim_frequency / control_freq)

        self.viewer.eye = (0.15, 0.25, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.02)

        active_joints = ["finger1_joint4", "finger2_joint4", "finger3_joint4"]
        passive_joints = [
            "finger1_joint1",
            "finger1_joint2",
            "finger1_joint3",
            "finger2_joint1",
            "finger2_joint2",
            "finger2_joint3",
            "finger3_joint1",
            "finger3_joint2",
            "finger3_joint3",
        ]


        self.scene.fingers = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UsdFileCfg(
                usd_path="/home/vsivtsov/Documents/kinova_isaac/3_fingers_matched_dimensions_mimic_joints.usd",
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.2),
                rot=(0.0, 1.0, 0.0, 0.0),
                joint_pos={ # values for slightly opeen position as in the Dorin's experiments
                    "finger1_joint1": -1.45 / 180 * pi,
                    "finger1_joint2": -14.47 / 180 * pi,
                    "finger1_joint3": -1.74 / 180 * pi,
                    "finger1_joint4": -1.74 / 180 * pi,

                    "finger2_joint1": -1.45 / 180 * pi,
                    "finger2_joint2": -14.47 / 180 * pi,
                    "finger2_joint3": -1.74 / 180 * pi,
                    "finger2_joint4": -1.74 / 180 * pi,

                    "finger3_joint1": -1.45 / 180 * pi,
                    "finger3_joint2": -14.47 / 180 * pi,
                    "finger3_joint3": -1.74 / 180 * pi,
                    "finger3_joint4": -1.74 / 180 * pi,
                },
            ),
            actuators={
                "finger1": mdp.AsymmetricActuatorCfg(
                    joint_names_expr=["finger1_joint4"],
                    stiffness=None,
                    damping=None,
                    close_taper=0.04,
                    max_range=ACTIVE_JOINT_RANGE_RAD,
                ),
                "finger2": mdp.AsymmetricActuatorCfg(
                    joint_names_expr=["finger2_joint4"],
                    stiffness=None,
                    damping=None,
                    close_taper=0.04,
                    max_range=ACTIVE_JOINT_RANGE_RAD,
                ),
                "finger3": mdp.AsymmetricActuatorCfg(
                    joint_names_expr=["finger3_joint4"],
                    stiffness=None,
                    damping=None,
                    close_taper=0.04,
                    max_range=ACTIVE_JOINT_RANGE_RAD,
                ),
                "passive": ImplicitActuatorCfg(
                    joint_names_expr=passive_joints,
                    stiffness=None,
                    damping=None,
                ),
            },
        )

        self.actions.finger_targets = mdp.JointPositionActionCfg(
            asset_name="fingers",
            scale=ACTIVE_JOINT_RANGE_RAD,
            offset=ACTIVE_JOINT_LOWER_RAD,
            use_default_offset=False,
            joint_names=active_joints,
            preserve_order=True,
        )
