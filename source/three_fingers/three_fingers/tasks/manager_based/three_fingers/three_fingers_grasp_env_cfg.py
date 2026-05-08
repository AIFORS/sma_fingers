# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from math import pi

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp


ACTIVE_JOINT_LOWER_DEG = -4.0
ACTIVE_JOINT_UPPER_DEG = 30.0
ACTIVE_JOINT_LOWER_RAD = ACTIVE_JOINT_LOWER_DEG / 180.0 * pi
ACTIVE_JOINT_UPPER_RAD = ACTIVE_JOINT_UPPER_DEG / 180.0 * pi
ACTIVE_JOINT_RANGE_RAD = ACTIVE_JOINT_UPPER_RAD - ACTIVE_JOINT_LOWER_RAD

SPHERE_BASE_DIAMETER = 0.065
SPHERE_DIAMETER_MIN = 0.014  # tested that radius 0.07 is the biggest that can be grasped when fingers are at the top hanging at 0.2m
SPHERE_DIAMETER_MAX = 0.07
SPHERE_SCALE_MIN = SPHERE_DIAMETER_MIN / SPHERE_BASE_DIAMETER
SPHERE_SCALE_MAX = SPHERE_DIAMETER_MAX / SPHERE_BASE_DIAMETER


@configclass
class ThreeFingersGraspSceneCfg(InteractiveSceneCfg):
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
            radius=SPHERE_BASE_DIAMETER / 2,
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, SPHERE_BASE_DIAMETER / 2)
        ),
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
class ThreeFingersGraspActionsCfg:
    finger_targets: mdp.JointPositionActionCfg = MISSING  # type: ignore[assignment]


@configclass
class ThreeFingersGraspObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        finger_state = ObsTerm(
            func=mdp.finger_state_normalized_from_open,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    joint_names=["finger1_joint4", "finger2_joint4", "finger3_joint4"],
                ),
                "lower_rad": ACTIVE_JOINT_LOWER_RAD,
                "upper_rad": ACTIVE_JOINT_UPPER_RAD,
            },
            history_length=6,
        )

        last_action = ObsTerm(func=mdp.last_action, history_length=6)

        sphere_size = ObsTerm(
            func=mdp.sphere_diameter,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    joint_names=[
                        "finger1_joint1",
                        "finger1_joint2",
                        "finger1_joint3",
                        "finger1_joint4",
                        "finger2_joint1",
                        "finger2_joint2",
                        "finger2_joint3",
                        "finger2_joint4",
                        "finger3_joint1",
                        "finger3_joint2",
                        "finger3_joint3",
                        "finger3_joint4",
                    ],
                )
            },
            history_length=6,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    joint_names=[
                        "finger1_joint1",
                        "finger1_joint2",
                        "finger1_joint3",
                        "finger1_joint4",
                        "finger2_joint1",
                        "finger2_joint2",
                        "finger2_joint3",
                        "finger2_joint4",
                        "finger3_joint1",
                        "finger3_joint2",
                        "finger3_joint3",
                        "finger3_joint4",
                    ],
                )
            },
            history_length=6,
        )
        sphere_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
            history_length=6,
        )
        sphere_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
            history_length=6,
        )
        sphere_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
            history_length=6,
        )
        sphere_size = ObsTerm(
            func=mdp.sphere_diameter,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )


        link5_pos = ObsTerm(
            func=mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    body_names=["link5"],
                    preserve_order=True,
                ),
            },
            history_length=6,
        )

        # link5-to-sphere vectors provide the fingertip->object displacement
        link5_to_sphere = ObsTerm(
            func=mdp.link5_to_sphere_pos,
            params={
                "fingers_asset_cfg": SceneEntityCfg(
                    "fingers",
                    body_names=["link5"],
                    preserve_order=True,
                ),
                "sphere_asset_cfg": SceneEntityCfg("sphere"),
            },
            history_length=6,
        )
        
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("fingers")},
            history_length=6,
        )
        contact_forces = ObsTerm(
            func=mdp.contact_forces_from_sensors,
            params={
                "sensor_names": [
                    "finger1_contact",
                    "finger2_contact",
                    "finger3_contact",
                ]
            },
            history_length=6,
        )

        net_sphere_force_mag_diff = ObsTerm(
            func=mdp.net_vs_sphere_contact_force_magnitude_diff,
            params={
                "sensor_names": [
                    "finger1_contact",
                    "finger2_contact",
                    "finger3_contact",
                ]
            },
            history_length=6,
        )

        contact_force_flags = ObsTerm(
            func=mdp.contact_force_flags,
            params={
                "sensor_names": [
                    "finger1_contact",
                    "finger2_contact",
                    "finger3_contact",
                ]
            },
            history_length=6,
        )

        contact_force_mid_diff = ObsTerm(
            func=mdp.contact_force_mid_diff,
            params={
                "sensor_names": [
                    "finger1_contact",
                    "finger2_contact",
                    "finger3_contact",
                ]
            },
            history_length=6,
        )

        last_action = ObsTerm(func=mdp.last_action, history_length=6)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class RndObjectCfg(ObsGroup):
        sphere_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )
        sphere_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )
        sphere_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class RndRobotCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    joint_names=[
                        "finger1_joint1",
                        "finger1_joint2",
                        "finger1_joint3",
                        "finger1_joint4",
                        "finger2_joint1",
                        "finger2_joint2",
                        "finger2_joint3",
                        "finger2_joint4",
                        "finger3_joint1",
                        "finger3_joint2",
                        "finger3_joint3",
                        "finger3_joint4",
                    ],
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "fingers",
                    joint_names=[
                        "finger1_joint1",
                        "finger1_joint2",
                        "finger1_joint3",
                        "finger1_joint4",
                        "finger2_joint1",
                        "finger2_joint2",
                        "finger2_joint3",
                        "finger2_joint4",
                        "finger3_joint1",
                        "finger3_joint2",
                        "finger3_joint3",
                        "finger3_joint4",
                    ],
                )
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    rnd_object: RndObjectCfg = RndObjectCfg()
    rnd_robot: RndRobotCfg = RndRobotCfg()


@configclass
class ThreeFingersGraspRewardsCfg:
    finger_symmetry = RewTerm(
        func=mdp.finger_symmetry,  # all finger states should be similar
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "fingers",
                joint_names=["finger1_joint4", "finger2_joint4", "finger3_joint4"],
            )
        },
    )

    tanh_finger_symmetry = RewTerm(
        func=mdp.tanh_finger_symmetry_reward,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "fingers",
                joint_names=["finger1_joint4", "finger2_joint4", "finger3_joint4"],
            ),
            "scale": 10.0,
        },
    )

    contact_balance = RewTerm(
        func=mdp.contact_force_balance,  # all contact forces should be similar
        weight=0.3,
        params={
            "sensor_names": ["finger1_contact", "finger2_contact", "finger3_contact"]
        },
    )

    is_touching = RewTerm(
        func=mdp.is_touching,  # each finger has contact force in proper range
        weight=0.2,
        params={
            "sensor_names": ["finger1_contact", "finger2_contact", "finger3_contact"]
        },
    )

    contact_forces_in_range = RewTerm(
        func=mdp.contact_force_in_range,  # finger 1 contact force is within [low, high]
        weight=8.0,
        params={
            "sensor_names": ["finger1_contact", "finger2_contact", "finger3_contact"]
        },
    )

    joint_efforts_in_range = RewTerm(
        func=mdp.joint_effort_all_in_range,  # all active-joint effort magnitudes are within [low, high]
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "fingers",
                joint_names=["finger1_joint4", "finger2_joint4", "finger3_joint4"],
            )
        },
    )

    contact_force_excess_penalty = RewTerm(
        func=mdp.contact_force_exceeds,  # number of fingers with contact force outside acceptable range
        weight=-2.0,
        params={
            "sensor_names": ["finger1_contact", "finger2_contact", "finger3_contact"]
        },
    )

    action_deviation = RewTerm(
        func=mdp.action_deviation,  # difference between target and actual joint positions
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "fingers",
                joint_names=["finger1_joint4", "finger2_joint4", "finger3_joint4"],
            ),
            "threshold": 0.8,
        },
    )

    opening_penalty = RewTerm(
        func=mdp.opening_penalty,  # penalize opening: target < current joint position
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "fingers",
                joint_names=["finger1_joint4", "finger2_joint4", "finger3_joint4"],
            ),
        },
    )

    xy_centering = RewTerm(
        func=mdp.sphere_xy_distance,  # sphere should be near the origin in XY plane
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("sphere")},
    )

    z_above_radius = RewTerm(
        func=mdp.sphere_z_above_radius,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("sphere")},
    )


    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)

    failure = RewTerm(
        func=mdp.is_terminated,
        weight=-20.0,
    )


@configclass
class ThreeFingersGraspTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    contact_force_exceeds_critical = DoneTerm(
        func=mdp.contact_force_exceeds_critical,
        params={
            "sensor_names": ["finger1_contact", "finger2_contact", "finger3_contact"]
        },
        time_out=False,
    )
    out_of_xy = DoneTerm(
        func=mdp.sphere_out_of_xy_bounds,
        params={"radius": 0.1, "asset_cfg": SceneEntityCfg("sphere")},
        time_out=False,
    )

@configclass
class ThreeFingersGraspEventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale_cached,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "scale_range": (SPHERE_SCALE_MIN, SPHERE_SCALE_MAX),
        },
    )
    reset_sphere = EventTerm(
        func=mdp.reset_sphere_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("sphere"), "spawn_cap": 0.001},
    )
    reset_fingers = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "fingers",
                joint_names=[
                    "finger1_joint1",
                    "finger1_joint2",
                    "finger1_joint3",
                    "finger1_joint4",
                    "finger2_joint1",
                    "finger2_joint2",
                    "finger2_joint3",
                    "finger2_joint4",
                    "finger3_joint1",
                    "finger3_joint2",
                    "finger3_joint3",
                    "finger3_joint4",
                ],
            ),
            "position_range": (0.9, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ThreeFingersGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: ThreeFingersGraspSceneCfg = ThreeFingersGraspSceneCfg(
        num_envs=4096, env_spacing=0.3
    )
    observations: ThreeFingersGraspObservationsCfg = ThreeFingersGraspObservationsCfg()
    actions: ThreeFingersGraspActionsCfg = ThreeFingersGraspActionsCfg()
    rewards: ThreeFingersGraspRewardsCfg = ThreeFingersGraspRewardsCfg()
    terminations: ThreeFingersGraspTerminationsCfg = ThreeFingersGraspTerminationsCfg()
    events: ThreeFingersGraspEventsCfg = ThreeFingersGraspEventsCfg()

    def __post_init__(self):
        # allow per-env USD edits (e.g., scale randomization)
        self.scene.replicate_physics = False

        sim_frequency = 100
        self.sim.dt = 1 / sim_frequency
        control_freq = 10
        self.decimation = int(sim_frequency / control_freq)
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0

        self.viewer.eye = (0.3, 0.35, 0.1)
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
                joint_pos={
                    "finger1_joint1": -3.0 / 180 * pi,
                    "finger1_joint2": -16.0 / 180 * pi,
                    "finger1_joint3": -4.0 / 180 * pi,
                    "finger1_joint4": -4.0 / 180 * pi,
                    "finger2_joint1": -3.0 / 180 * pi,
                    "finger2_joint2": -16.0 / 180 * pi,
                    "finger2_joint3": -4.0 / 180 * pi,
                    "finger2_joint4": -4.0 / 180 * pi,
                    "finger3_joint1": -3.0 / 180 * pi,
                    "finger3_joint2": -16.0 / 180 * pi,
                    "finger3_joint3": -4.0 / 180 * pi,
                    "finger3_joint4": -4.0 / 180 * pi,
                },
            ),
            actuators={
                "finger": mdp.AsymmetricActuatorCfg(
                    joint_names_expr=active_joints,
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
