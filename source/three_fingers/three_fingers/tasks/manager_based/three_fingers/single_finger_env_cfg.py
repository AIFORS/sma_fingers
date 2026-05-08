# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

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


@configclass
class SingleFingerSceneCfg(InteractiveSceneCfg):
    """Scene with a single finger mechanism."""

    fingers: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=None),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    """

    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.065,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    sphere_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/fingers_on_base/finger_1/link5"],
    )
    """


@configclass
class SingleFingerActionsCfg:
    finger_targets: mdp.JointPositionActionCfg = MISSING  # type: ignore[assignment]


@configclass
class SingleFingerObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            scale=1.0 / 0.523598776,
            params={
                "asset_cfg": SceneEntityCfg("fingers", joint_names=["finger1_joint4"])
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class SingleFingerEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the single finger joint-target environment."""

    scene: SingleFingerSceneCfg = SingleFingerSceneCfg(num_envs=1, env_spacing=1.5)
    observations: SingleFingerObservationsCfg = SingleFingerObservationsCfg()
    actions: SingleFingerActionsCfg = SingleFingerActionsCfg()

    def __post_init__(self):
        sim_frequency = 100  # simulation frequency in Hz
        self.sim.dt = 1 / sim_frequency

        control_freq = 10  # control frequency -> Hz
        self.decimation = int(sim_frequency / control_freq)

        # self.sim.render_interval = self.decimation
        self.viewer.eye = (1.0, 1.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.7)

        active_joints = ["finger1_joint4"]
        passive_joints = ["finger1_joint1", "finger1_joint2", "finger1_joint3"]

        self.scene.fingers = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UsdFileCfg(
                usd_path="C:/Users/calculated/Documents/Projects/isaacGym/IsaacLab/3_fingers_drive_at_4_joint.usd",
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.2),
                joint_pos={name: 0.0 for name in active_joints + passive_joints},
            ),
            actuators={
                "finger": mdp.AsymmetricActuatorCfg(
                    joint_names_expr=active_joints,
                    stiffness=10.0,
                    damping=6.0,
                    close_taper=0.22,
                ),
                "passive": ImplicitActuatorCfg(
                    joint_names_expr=passive_joints,
                    stiffness=0.0,
                    damping=0.0,
                ),
            },
        )

        self.actions.finger_targets = mdp.JointPositionActionCfg(
            asset_name="fingers",
            scale=0.523598776,  # 30 degrees in radians
            joint_names=active_joints,
            preserve_order=True,
        )
