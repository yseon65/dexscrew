# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Isaac Lab port of dexscrew/tasks/xhand_hora.py (originally written for IsaacGym Preview 4).
#
# The task logic (rewards, terminations, observations, privileged info, domain randomization)
# follows the IsaacGym implementation line by line. Internally all quaternions keep the
# IsaacGym (x, y, z, w) convention and all positions are expressed in the env-local frame,
# exactly like the IsaacGym root/rigid-body state tensors. Conversion to Isaac Lab
# (w, x, y, z quaternions, world frame) happens only when reading from / writing to the sim.
# --------------------------------------------------------

import os
import random
from collections import OrderedDict
from glob import glob

import gymnasium as gym
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from pxr import PhysxSchema, UsdPhysics

from dexscrew.utils.misc import tprint
from dexscrew.utils.point_cloud_prep import sample_cylinder
from dexscrew.utils.torch_jit_utils import (
    quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul, tensor_clamp, to_torch, torch_rand_float,
    wxyz_to_xyzw, xyzw_to_wxyz,
)

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
USD_DIR = os.path.join(REPO_ROOT, 'assets', 'usd')

# IsaacGym orders the DOFs of the hand alphabetically; all task buffers use this order.
HAND_DOF_NAMES = [
    'left_hand_index_bend_joint', 'left_hand_index_joint1', 'left_hand_index_joint2',
    'left_hand_mid_joint1', 'left_hand_mid_joint2', 'left_hand_pinky_joint1', 'left_hand_pinky_joint2',
    'left_hand_ring_joint1', 'left_hand_ring_joint2', 'left_hand_thumb_bend_joint',
    'left_hand_thumb_rota_joint1', 'left_hand_thumb_rota_joint2',
]
FINGERTIP_NAMES = ['left_hand_index_rota_tip', 'left_hand_mid_tip', 'left_hand_pinky_tip',
                   'left_hand_ring_tip', 'left_hand_thumb_rota_tip']
# IsaacGym rigid body order of the screw articulation: base (root), bolt, nut (the revolving link).
# The screwdriver assets name them base / shaft / handle, the nut-bolt assets base / bolt / nut.
OBJECT_BODY_NAMES = {
    'driver': ['base', 'shaft', 'handle'],
    'default': ['base', 'bolt', 'nut'],
}


def object_body_names(object_type):
    subset_name = object_type.split('_')[-1] if '_' in object_type else 'None'
    return OBJECT_BODY_NAMES.get(subset_name, OBJECT_BODY_NAMES['default'])

# IsaacGym AssetOptions defaults that the original code relied on (max_angular_velocity is in rad/s there)
GYM_MAX_ANGULAR_VELOCITY_DEG = float(np.rad2deg(64.0))


def _rigid_props(disable_gravity, angular_damping):
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        linear_damping=0.0,
        angular_damping=angular_damping,
        max_linear_velocity=1000.0,
        max_angular_velocity=GYM_MAX_ANGULAR_VELOCITY_DEG,
        max_depenetration_velocity=1000.0,
    )


def _articulation_props():
    return sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        fix_root_link=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.0,
        stabilization_threshold=0.0,
    )


def build_lab_cfg(config, sim_device, headless):
    """Build the Isaac Lab env cfg from the (hydra) task config dict of the original code."""
    env_cfg = config['env']
    num_envs = env_cfg['numEnvs']
    rand_cfg = env_cfg['randomization']
    sim_cfg = config['sim']

    # ---- object assets: env i gets scale bucket (i % num_scales), like the IsaacGym code
    object_type = env_cfg['object']['type']
    subset_name = object_type.split('_')[-1] if '_' in object_type else 'None'
    urdfs = sorted(glob(os.path.join(REPO_ROOT, f'assets/screw/{subset_name}/*.urdf')))
    assert len(urdfs) > 0, f'no screw assets found for {object_type}'
    if rand_cfg['randomizeScale']:
        scale_list = list(rand_cfg['randomizeScaleList'])
    else:
        scale_list = [env_cfg['baseObjScale']]
    asset_cfgs, asset_meta = [], []
    # asset index k -> scale scale_list[k % S], object type k // S; with MultiAssetSpawner(random_choice=False)
    # env i is assigned asset i % (S * T), hence its scale is scale_list[i % S].
    for type_id, urdf in enumerate(urdfs):
        name = os.path.splitext(os.path.basename(urdf))[0]
        usd = os.path.join(USD_DIR, subset_name, name, f'{name}.usda')
        assert os.path.exists(usd), f'{usd} missing, run scripts/convert_urdf_to_usd.py first'
        for s in scale_list:
            asset_cfgs.append(sim_utils.UsdFileCfg(usd_path=usd, scale=(s, s, s), articulation_props=_articulation_props()))
            asset_meta.append((type_id, s, urdf))

    base_name, bolt_name, nut_name = object_body_names(object_type)
    nut_prim = f'{{ENV_REGEX_NS}}/Object/Geometry/{base_name}/{bolt_name}/{nut_name}'

    hand_usd = os.path.join(USD_DIR, 'xhand_left', 'xhand_left.usda')
    assert os.path.exists(hand_usd), f'{hand_usd} missing, run scripts/convert_urdf_to_usd.py first'

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path='/World/light',
                             spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
        hand = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Hand',
            spawn=sim_utils.UsdFileCfg(
                usd_path=hand_usd,
                rigid_props=_rigid_props(disable_gravity=True, angular_damping=0.01),
                articulation_props=_articulation_props(),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.21)),
            actuators={'hand': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0.0, damping=0.0)},
        )
        object = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Object',
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=asset_cfgs,
                random_choice=False,
                rigid_props=_rigid_props(disable_gravity=False, angular_damping=0.5),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.21)),
            actuators={'screw': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0.0, damping=0.0)},
        )
        handle_contact = ContactSensorCfg(prim_path=nut_prim,
                                          update_period=0.0, history_length=0)

    physx = sim_cfg['physx']
    # GPU buffers sized for the full 8192-env training run; small runs (e.g. a live viewer next to a
    # training job) use 1/16 of that so they fit in the remaining GPU memory.
    shrink = 16 if num_envs <= 256 else 1
    gpu_buffers = dict(
        gpu_max_rigid_contact_count=physx['max_gpu_contact_pairs'] // shrink,
        gpu_max_rigid_patch_count=physx['max_gpu_contact_pairs'] // 4 // shrink,
        gpu_found_lost_pairs_capacity=2 ** 23 // shrink,
        gpu_found_lost_aggregate_pairs_capacity=2 ** 25 // shrink,
        gpu_total_aggregate_pairs_capacity=2 ** 23 // shrink,
        gpu_heap_capacity=2 ** 26 // shrink,
        gpu_temp_buffer_capacity=2 ** 24 // shrink,
    )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = env_cfg['controller']['controlFrequencyInv']
        episode_length_s = env_cfg['episodeLength'] * sim_cfg['dt'] * env_cfg['controller']['controlFrequencyInv']
        action_space = env_cfg['numActions']
        observation_space = 32 * 3
        state_space = 0
        sim: SimulationCfg = SimulationCfg(
            dt=sim_cfg['dt'],
            render_interval=env_cfg['controller']['controlFrequencyInv'],
            device=sim_device,
            gravity=tuple(sim_cfg['gravity']),
            physx=PhysxCfg(
                solver_type=physx['solver_type'],
                min_position_iteration_count=1,
                max_position_iteration_count=255,
                bounce_threshold_velocity=physx['bounce_threshold_velocity'],
                **gpu_buffers,
            ),
        )
        scene: InteractiveSceneCfg = SceneCfg(num_envs=num_envs, env_spacing=env_cfg['envSpacing'],
                                              replicate_physics=False)

    cfg = EnvCfg()
    cfg.seed = config.get('seed', None)
    # GUI camera: close-up of env 0 (hand at z=0.21, nut/handle around z=0.1)
    cfg.viewer.origin_type = 'env'
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.25, 0.35, 0.30)
    cfg.viewer.lookat = (0.0, 0.05, 0.10)
    return cfg, asset_meta


class XHandHoraLab(DirectRLEnv):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        self.numActions = config['env']['numActions']
        self.num_actions = self.numActions
        self.fingers_num = 5
        self._setup_domain_rand_config(config['env']['randomization'])
        self._setup_priv_option_config(config['env']['privInfo'])
        self._setup_object_info(config['env']['object'])
        self._setup_reward_config(config['env']['reward'])
        self.base_obj_scale = config['env']['baseObjScale']
        self.up_axis = 'z'
        self.rotation_axis = config['env']['rotation_axis']
        self.reset_z_threshold = self.config['env']['reset_z_threshold']
        self.reset_dist_threshold = self.config['env']['reset_dist_threshold']
        self.with_camera = False
        self.enable_camera_sensors = False
        self.nut_termination_history_len = config['env']['object']['nut_termination_history_len']
        self.nut_stagnation_eps = config['env']['object']['nut_stagnation_eps']
        self.screw_upper_limit = config['env']['object']['screw_upper_limit']
        self.headless = headless

        # ---- VecTask attributes
        self.rl_device = sim_device
        self.num_observations = 32 * 3
        self.clip_obs = config['env'].get('clipObservations', np.inf)
        self.clip_actions = config['env'].get('clipActions', np.inf)
        controller_config = config['env']['controller']
        self.torque_control = controller_config['torque_control']
        assert self.torque_control, 'only torque control is ported'
        self.p_gain = controller_config['pgain']
        self.d_gain = controller_config['dgain']
        self.control_freq_inv = controller_config['controlFrequencyInv']
        self.action_scale = controller_config['action_scale']
        self.torque_limit = controller_config['torque_limit']

        lab_cfg, self.object_asset_meta = build_lab_cfg(config, sim_device, headless)
        super().__init__(lab_cfg)

        # gym spaces as in VecTask (un-batched)
        self.obs_space = gym.spaces.Box(np.ones(self.num_obs, dtype=np.float32) * -np.inf,
                                        np.ones(self.num_obs, dtype=np.float32) * np.inf)
        self.act_space = gym.spaces.Box(np.ones(self.num_actions, dtype=np.float32) * -1.,
                                        np.ones(self.num_actions, dtype=np.float32) * 1.)
        self.observation_space = self.obs_space
        self.action_space = self.act_space

        self.hand = self.scene['hand']
        self.object = self.scene['object']
        self.handle_contact = self.scene['handle_contact']
        self.env_origins = self.scene.env_origins

        self._allocate_buffers()
        self._setup_asset_handles()
        self._setup_asset_properties()

        self.eval_done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # DirectRLEnv.max_episode_length = episode_length_s / step_dt = episodeLength
        assert self.max_episode_length == self.config['env']['episodeLength']
        self.dt = self.physics_dt
        self._sim_frame_count = 0

        self.xhand_hand_default_dof_pos = torch.zeros(self.num_xhand_hand_dofs, dtype=torch.float, device=self.device)
        self.pose_diff_penalty_mask = torch.ones(self.num_actions, device=self.device, dtype=torch.float)
        thumb_indices = [i for i, name in enumerate(HAND_DOF_NAMES) if 'thumb' in name]
        self.pose_diff_penalty_mask[thumb_indices] = 0.0

        self._refresh_gym()

        self.num_dofs = self.num_xhand_hand_dofs + 1  # hand + screw joint
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, len(self.object_body_ids), 3), dtype=torch.float, device=self.device)
        self.contact_thresh = torch.zeros((self.num_envs, self.num_contacts), dtype=torch.float, device=self.device)

        self.saved_grasping_states = {}
        num_random_poses = 5000
        if self.config['env']['initPose'] == 'nutbolt_inclined':
            joint_values = {
                "left_hand_index_bend_joint": -0.17,
                "left_hand_index_joint1": 1.1,
                "left_hand_index_joint2": 0.4,
                "left_hand_mid_joint1": 1.1,
                "left_hand_mid_joint2": 0.4,
                "left_hand_pinky_joint1": 0,
                "left_hand_pinky_joint2": 0,
                "left_hand_ring_joint1": 0,
                "left_hand_ring_joint2": 0,
                "left_hand_thumb_bend_joint": 1.3,
                "left_hand_thumb_rota_joint1": 0.5,
                "left_hand_thumb_rota_joint2": 0.45,
            }
        elif self.config['env']['initPose'] == 'screwdriver_inclined':
            joint_values = {
                "left_hand_index_bend_joint": -0.036,
                "left_hand_index_joint1": 1.15,
                "left_hand_index_joint2": 0.5,
                "left_hand_mid_joint1": 0.925,
                "left_hand_mid_joint2": 0.58,
                "left_hand_pinky_joint1": 0,
                "left_hand_pinky_joint2": 0,
                "left_hand_ring_joint1": 1.3,
                "left_hand_ring_joint2": 0.43,
                "left_hand_thumb_bend_joint": 1.455,
                "left_hand_thumb_rota_joint1": 0.817,
                "left_hand_thumb_rota_joint2": 0.154,
            }
        assert HAND_DOF_NAMES == list(joint_values.keys())
        self.joint_values_lst = list(joint_values.values())
        for s in self.randomize_scale_list:
            scale_key = str(s)
            random_pose_data = torch.zeros((num_random_poses, 19), device=self.device, dtype=torch.float)
            for i in range(12):
                random_pose_data[:, i] = torch.ones(num_random_poses, device=self.device) * self.joint_values_lst[i]
            random_pose_data[:, 14] = self.reset_z_threshold + 0.1
            random_pose_data[:, 18] = 1.0
            self.saved_grasping_states[scale_key] = random_pose_data

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        sign, axis = self.rotation_axis[0], self.rotation_axis[1]
        axis_index = ['x', 'y', 'z'].index(axis)
        self.rot_axis_buf[:, axis_index] = 1
        self.rot_axis_buf[:, axis_index] = -self.rot_axis_buf[:, axis_index] if sign == '-' else self.rot_axis_buf[:, axis_index]

        # useful buffers
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.control_freq_inv, self.num_dofs), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.control_freq_inv, self.num_actions), device=self.device, dtype=torch.float)

        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        self.ft_pos_prev = self.fingertip_pos.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()
        self.nut_dof_pos_prev = self.nut_dof_pos.clone()

        self.obj_linvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_angvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.ft_linvel_at_cf = torch.zeros((self.num_envs, self.fingers_num * 3), device=self.device, dtype=torch.float)
        self.ft_angvel_at_cf = torch.zeros((self.num_envs, self.fingers_num * 3), device=self.device, dtype=torch.float)
        self.nut_dof_vel_cf = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.dof_acc = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        # debug and statistics
        self.evaluate = self.config['on_evaluation']
        self.evaluate_cache_name = self.config['eval_cache_name']
        self.stat_sum_rewards = [0 for _ in self.object_type_list]
        self.stat_sum_episode_length = [0 for _ in self.object_type_list]
        self.stat_sum_rotate_rewards = [0 for _ in self.object_type_list]
        self.stat_sum_rotate_penalty = [0 for _ in self.object_type_list]
        self.stat_sum_unclip_rotate_rewards = [0 for _ in self.object_type_list]
        self.stat_sum_unclip_rotate_penalty = [0 for _ in self.object_type_list]
        self.extrin_log = []
        self.env_evaluated = [0 for _ in self.object_type_list]
        self.evaluate_iter = 0
        self.obs_dict = {}

    # ------------------------------------------------------------------
    # VecTask-like properties
    # ------------------------------------------------------------------
    @property
    def num_obs(self):
        return self.num_observations

    @property
    def num_acts(self):
        return self.num_actions

    def _allocate_buffers(self):
        num_envs = self.num_envs
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.obs_buf_lag_history = torch.zeros((num_envs, 80, self.numActions * 2), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(num_envs, device=self.device, dtype=torch.long)
        self.at_reset_buf = torch.ones(num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        self._allocate_task_buffer(num_envs)

    # ------------------------------------------------------------------
    # Scene setup (replaces _create_envs / _create_object_asset / _parse_hand_dof_props)
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # The screw links are nested rigid bodies, so `activate_contact_sensors` would only tag the root body.
        # Tag every object body explicitly so the handle ("nut") can report its net contact force.
        stage = self.scene.stage
        for env_id in range(self.scene.num_envs):
            root = stage.GetPrimAtPath(f'{self.scene.env_prim_paths[env_id]}/Object')
            stack = [root]
            while stack:
                prim = stack.pop()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                    cr_api.CreateThresholdAttr().Set(0.0)
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr().Set(0.0)
                stack += list(prim.GetChildren())

    def _setup_asset_handles(self):
        self.hand_dof_ids = self.hand.find_joints(HAND_DOF_NAMES, preserve_order=True)[0]
        self.num_xhand_hand_dofs = len(self.hand_dof_ids)
        assert self.num_xhand_hand_dofs == self.numActions
        self.hand_dof_ids_t = torch.tensor(self.hand_dof_ids, device=self.device, dtype=torch.long)
        self.fingertip_handles = self.hand.find_bodies(FINGERTIP_NAMES, preserve_order=True)[0]
        body_names = object_body_names(self.object_type)
        self.object_body_ids = self.object.find_bodies(body_names, preserve_order=True)[0]
        self.screw_base_rb_handle, self.screw_bolt_rb_handle, self.screw_nut_rb_handle = self.object_body_ids
        self.object_rb_handles = to_torch(self.object_body_ids, dtype=torch.long, device=self.device)
        assert self.object.num_joints == 1
        self.handle_contact_body = self.handle_contact.find_bodies(body_names[-1])[0][0]

    def _setup_asset_properties(self):
        """Joint properties (from _parse_hand_dof_props) and per-env domain randomization (from _create_envs)."""
        num_envs = self.num_envs
        all_env_ids = torch.arange(num_envs, device=self.device)

        # ---- hand DOF properties
        xhand_dof_lower_limits = [-0.175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.05, -0.17]
        xhand_dof_upper_limits = [0.175, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.83, 1.57, 1.83]
        if self.config['env']['object']['thumb_range_limit']:
            xhand_dof_upper_limits[9] = 1.73
            xhand_dof_lower_limits[9] = 0.6
        self.xhand_dof_lower_limits = np.array(xhand_dof_lower_limits)
        self.xhand_dof_upper_limits = np.array(xhand_dof_upper_limits)
        # effort limits as indexed in the original code (applied in IsaacGym DOF order)
        xhand_effort_limits = [0.4, 1.1, 0.4, 1.1, 0.4, 1.1, 0.4, 1.1, 1.1, 0.4, 1.1, 1.1, 0.4]
        self.xhand_hand_dof_lower_limits = to_torch(xhand_dof_lower_limits, device=self.device)
        self.xhand_hand_dof_upper_limits = to_torch(xhand_dof_upper_limits, device=self.device)
        self.xhand_effort_limits = to_torch(xhand_effort_limits[:self.num_xhand_hand_dofs], device=self.device)

        limits = torch.stack([self.xhand_hand_dof_lower_limits, self.xhand_hand_dof_upper_limits], dim=-1)
        self.hand.write_joint_position_limit_to_sim(limits[None].repeat(num_envs, 1, 1), joint_ids=self.hand_dof_ids,
                                                    warn_limit_violation=False)
        self.hand.write_joint_effort_limit_to_sim(self.xhand_effort_limits[None].repeat(num_envs, 1), joint_ids=self.hand_dof_ids)
        self.hand.write_joint_friction_coefficient_to_sim(0.01, joint_dynamic_friction_coeff=0.01, joint_ids=self.hand_dof_ids)
        self.hand.write_joint_armature_to_sim(0.001, joint_ids=self.hand_dof_ids)
        # screw joint: URDF <dynamics friction="0.2"/>
        self.object.write_joint_friction_coefficient_to_sim(0.2, joint_dynamic_friction_coeff=0.2)
        self.object.write_joint_armature_to_sim(0.0)

        # ---- per-env object type / scale (fixed at spawn by the MultiAssetSpawner)
        n_assets = len(self.object_asset_meta)
        object_type_at_env, obj_scales, point_clouds = [], [], []
        asset_point_clouds = {}
        for i in range(num_envs):
            type_id, scale, urdf = self.object_asset_meta[i % n_assets]
            object_type_at_env.append(type_id)
            obj_scales.append(scale)
            if type_id not in asset_point_clouds:
                asset_point_clouds[type_id] = self._load_point_cloud(urdf)
            if self.point_cloud_sampled_dim > 0:
                point_clouds.append(asset_point_clouds[type_id] * scale)
        self.object_type_at_env = to_torch(object_type_at_env, dtype=torch.long, device=self.device)
        self.obj_point_clouds = to_torch(np.array(point_clouds), device=self.device, dtype=torch.float)
        obj_scales = to_torch(obj_scales, device=self.device)
        self._update_priv_buf(env_id=all_env_ids, name='screw_joint_friction', value=0.2)
        self._update_priv_buf(env_id=all_env_ids, name='obj_scale', value=obj_scales[:, None])

        cpu_ids = torch.arange(num_envs, dtype=torch.int32)
        obj_view = self.object.root_physx_view
        hand_view = self.hand.root_physx_view
        base_id, nut_id = self.screw_base_rb_handle, self.screw_nut_rb_handle

        # ---- COM randomization
        obj_com = torch.zeros((num_envs, 3))
        if self.randomize_com:
            coms = obj_view.get_coms().clone()
            lo, hi = self.randomize_com_lower, self.randomize_com_upper
            base_com = torch.rand((num_envs, 3)) * (hi - lo) / 2 + lo / 2
            nut_com = torch.rand((num_envs, 3)) * (hi - lo) + lo
            nut_com[:, 2] += 0.35
            coms[:, base_id, :3] = base_com
            coms[:, nut_id, :3] = nut_com
            obj_view.set_coms(coms, cpu_ids)
            obj_com = base_com
        self._update_priv_buf(env_id=all_env_ids, name='obj_com', value=obj_com.to(self.device))

        # ---- friction / restitution (same values for all hand and object shapes of an env)
        obj_friction = torch.ones(num_envs)
        obj_restitution = torch.zeros(num_envs)
        if self.randomize_friction:
            obj_friction = torch.rand(num_envs) * (self.randomize_friction_upper - self.randomize_friction_lower) + self.randomize_friction_lower
            obj_restitution = torch.rand(num_envs)
            for view in [hand_view, obj_view]:
                mat = view.get_material_properties().clone()
                mat[..., 0] = obj_friction[:, None]
                mat[..., 1] = obj_friction[:, None]
                mat[..., 2] = obj_restitution[:, None]
                view.set_material_properties(mat, cpu_ids)
        self._update_priv_buf(env_id=all_env_ids, name='obj_friction', value=obj_friction[:, None].to(self.device))
        self._update_priv_buf(env_id=all_env_ids, name='obj_restitution', value=obj_restitution[:, None].to(self.device))

        # ---- mass
        masses = obj_view.get_masses().clone()
        if self.randomize_mass:
            masses = torch.rand(masses.shape) * (self.randomize_mass_upper - self.randomize_mass_lower) + self.randomize_mass_lower
            obj_view.set_masses(masses, cpu_ids)
        self.obj_mass = masses[:, base_id].to(self.device)
        self._update_priv_buf(env_id=all_env_ids, name='obj_mass', value=self.obj_mass[:, None])

    def _load_point_cloud(self, urdf):
        pc_file = urdf.replace('.urdf', '.npy')
        if os.path.exists(pc_file):
            screw_points = np.load(pc_file)
            if len(screw_points) != self.point_cloud_sampled_dim:
                replace = len(screw_points) < self.point_cloud_sampled_dim
                indices = np.random.choice(len(screw_points), self.point_cloud_sampled_dim, replace=replace)
                screw_points = screw_points[indices]
            return screw_points
        return sample_cylinder(1.0) * 0.022

    # ------------------------------------------------------------------
    # stepping (VecTask.step + XHandHora.step)
    # ------------------------------------------------------------------
    def reset(self):
        env_ids = self.reset_buf.nonzero().squeeze(-1)
        self.reset_idx(env_ids)
        zero_actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)
        self.step(zero_actions)
        self.obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return self.obs_dict

    def step(self, actions, extrin_record=None):
        action_mask = torch.ones_like(actions)
        if self.config['env']['initPose'] == 'screwdriver_inclined':
            action_mask[:, 5:7] = 0.0
        else:
            action_mask[:, 5:9] = 0.0
        actions = actions * action_mask
        actions = F.pad(actions, (0, 1), value=0.0)  # pad the last dim with 0.0 for the nut joint

        if extrin_record is not None and self.config['env']['object']['evalObjectType'] is not None:
            self.extrin_log.append(
                (extrin_record.detach().cpu().numpy().copy(), self.eval_done_buf.detach().cpu().numpy().copy())
            )

        # ---- VecTask.step
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.action_saved = action_tensor.clone()
        self.pre_physics_step(action_tensor)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        force_indices = random.choices(list(range(self.control_freq_inv)), k=2)
        for i in range(self.control_freq_inv):
            self.update_low_level_control(i)
            if i in force_indices:
                self.update_rigid_body_force()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._sim_frame_count += 1
            if is_rendering and self._sim_frame_count % self.cfg.sim.render_interval == 0:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.timeout_buf = torch.where(
            torch.greater_equal(self.progress_buf, self.max_episode_length - 1),
            torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf)
        )
        self.post_physics_step()
        self.extras['time_outs'] = self.timeout_buf.to(self.rl_device)
        self.obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # ---- XHandHora.step
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        targets = self.prev_targets + self.action_scale * self.actions
        self.cur_targets[:, :self.num_xhand_hand_dofs] = tensor_clamp(
            targets[:, :self.num_xhand_hand_dofs], self.xhand_hand_dof_lower_limits, self.xhand_hand_dof_upper_limits)

        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.nut_dof_pos_prev[:] = self.nut_dof_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        self.ft_pos_prev[:] = self.fingertip_pos
        self.dof_vel_prev[:] = self.dof_vel_finite_diff

    def update_low_level_control(self, step_id):
        previous_dof_pos = self.xhand_hand_dof_pos.clone()
        self._refresh_gym()
        random_action_noise_t = torch.normal(0, self.random_action_noise_t_scale, size=self.xhand_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noise_action = self.cur_targets[..., :self.num_xhand_hand_dofs] + self.random_action_noise_e + random_action_noise_t

        dof_pos = self.xhand_hand_dof_pos
        dof_vel = (dof_pos - previous_dof_pos) / self.dt
        self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
        torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
        torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
        self.torques[:, step_id, :self.num_xhand_hand_dofs] = torques.clone()
        # IsaacGym clamps DOF_MODE_EFFORT actuation forces with the DOF effort limit
        applied = torch.max(torch.min(torques, self.xhand_effort_limits), -self.xhand_effort_limits)
        self.hand.set_joint_effort_target(applied, joint_ids=self.hand_dof_ids)
        self.object.set_joint_effort_target(self.torques[:, step_id, self.num_xhand_hand_dofs:])

    def update_rigid_body_force(self):
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero().squeeze(-1)
            self.rb_forces[force_indices] = torch.randn(
                self.rb_forces[force_indices].shape,
                device=self.device) * self.obj_mass[force_indices, None, None] * self.force_scale
            # forces act on the next simulation step only (like apply_rigid_body_force_tensors)
            self.object.instantaneous_wrench_composer.add_forces_and_torques(
                forces=self.rb_forces, torques=torch.zeros_like(self.rb_forces),
                body_ids=self.object_rb_handles, is_global=True)

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

    # ------------------------------------------------------------------
    # state access (replaces IsaacGym tensor views)
    # ------------------------------------------------------------------
    def _refresh_gym(self):
        hand_data, obj_data = self.hand.data, self.object.data
        origins = self.env_origins
        self.xhand_hand_dof_pos = hand_data.joint_pos[:, self.hand_dof_ids_t].clone()
        self.xhand_hand_dof_vel = hand_data.joint_vel[:, self.hand_dof_ids_t].clone()

        self.object_pos = obj_data.root_link_pos_w - origins
        self.object_rot = wxyz_to_xyzw(obj_data.root_link_quat_w)
        self.object_pose = torch.cat([self.object_pos, self.object_rot], dim=-1)
        self.object_linvel = obj_data.root_com_lin_vel_w.clone()
        self.object_angvel = obj_data.root_com_ang_vel_w.clone()

        ft_ids = self.fingertip_handles
        self.fingertip_pos = (hand_data.body_link_pos_w[:, ft_ids] - origins[:, None]).reshape(self.num_envs, -1)
        self.fingertip_orientation = wxyz_to_xyzw(hand_data.body_link_quat_w[:, ft_ids]).reshape(self.num_envs, -1)
        self.fingertip_linvel = hand_data.body_com_lin_vel_w[:, ft_ids].reshape(self.num_envs, -1)
        self.fingertip_angvel = hand_data.body_com_ang_vel_w[:, ft_ids].reshape(self.num_envs, -1)

        self.nut_pos = obj_data.body_link_pos_w[:, self.screw_nut_rb_handle] - origins
        forces = self.handle_contact.data.net_forces_w[:, self.handle_contact_body]
        self.nut_contact = torch.norm(forces, dim=-1).unsqueeze(-1)
        self.nut_dof_pos = obj_data.joint_pos[:, 0:1].clone()
        self.nut_dof_vel = obj_data.joint_vel[:, 0:1].clone()

    def _thumb_index_pos(self):
        ft = self.fingertip_pos.view(self.num_envs, self.fingers_num, 3)
        return ft[:, -1], ft[:, 0]

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            pgain_priv = torch.zeros((len(env_ids), self.num_xhand_hand_dofs), device=self.device, dtype=torch.float)
            dgain_priv = torch.zeros((len(env_ids), self.num_xhand_hand_dofs), device=self.device, dtype=torch.float)
            for joint_idx in range(self.num_xhand_hand_dofs):
                pgain_priv[:, joint_idx] = torch_rand_float(
                    self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 1), device=self.device).squeeze(-1)
                dgain_priv[:, joint_idx] = torch_rand_float(
                    self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), 1), device=self.device).squeeze(-1)
                self.p_gain[env_ids, joint_idx] = pgain_priv[:, joint_idx]
                self.d_gain[env_ids, joint_idx] = dgain_priv[:, joint_idx]
            self._update_priv_buf(env_ids, 'pgain', pgain_priv)
            self._update_priv_buf(env_ids, 'dgain', dgain_priv)

        self.random_obs_noise_e[env_ids] = torch.normal(0, self.random_obs_noise_e_scale, size=(len(env_ids), self.num_actions), device=self.device, dtype=torch.float)
        self.random_action_noise_e[env_ids] = torch.normal(0, self.random_action_noise_e_scale, size=(len(env_ids), self.num_actions), device=self.device, dtype=torch.float)
        self.rb_forces[env_ids, :, :] = 0.0

        n = len(env_ids)
        object_root = torch.zeros((n, 7), device=self.device, dtype=torch.float)  # local pos + xyzw quat
        hand_dof_pos = torch.zeros((n, self.num_xhand_hand_dofs), device=self.device, dtype=torch.float)
        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            sel = (env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)
            s_ids = env_ids[sel]
            if len(s_ids) == 0:
                continue
            scale_key = str(self.randomize_scale_list[n_s])
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            random_noise_x, random_noise_y = np.random.uniform(0, 0.0075), np.random.uniform(0, 0.0075)
            if self.config['env']['initPose'] == 'nutbolt_inclined':
                self.object_x, self.object_y, self.object_z = 0.0175 + random_noise_x, 0.06 + random_noise_y, 0
            elif self.config['env']['initPose'] == 'screwdriver_inclined':
                self.object_x, self.object_y, self.object_z = 0.009 + random_noise_x, 0.06 + random_noise_y, 0

            sampled_pose[:, self.numActions + 0] = self.object_x
            sampled_pose[:, self.numActions + 1] = self.object_y
            sampled_pose[:, self.numActions + 2] = self.object_z

            object_pose_noise = torch.normal(0, self.random_pose_noise, size=(sampled_pose.shape[0], 7), device=self.device, dtype=torch.float)
            object_pose_noise[:, 3:] = 0
            object_root[sel] = sampled_pose[:, self.numActions:] + object_pose_noise

            pos = sampled_pose[:, :self.numActions]
            self._update_priv_buf(env_id=s_ids, name='hand_joint_pos', value=pos)
            half_widths = torch.tensor(0.1 * (self.xhand_dof_upper_limits - self.xhand_dof_lower_limits) / 2.0, device=self.device, dtype=torch.float)
            hand_joint_noise = (torch.rand_like(half_widths) * 2 - 1) * half_widths
            pos = pos + hand_joint_noise
            self.sampled_pose = sampled_pose[:, :self.numActions]
            hand_dof_pos[sel] = pos
            self.prev_targets[s_ids, :self.num_xhand_hand_dofs] = pos
            self.cur_targets[s_ids, :self.num_xhand_hand_dofs] = pos

            screw_joint_init = torch.zeros((len(s_ids), self.num_dofs - self.num_xhand_hand_dofs), device=self.device)
            self.prev_targets[s_ids, self.num_xhand_hand_dofs:] = screw_joint_init
            self.cur_targets[s_ids, self.num_xhand_hand_dofs:] = screw_joint_init
            self.init_pose_buf[s_ids, :] = torch.cat([pos, screw_joint_init], dim=1)

        # hand orientation: X-axis rotation (20-30 deg) and Y-axis rotation (-5..5 deg)
        random_degrees_x = torch.rand(n, device=self.device) * 10 + 20
        angles_x = np.pi / 2 - random_degrees_x * (np.pi / 180)
        cos_half_x, sin_half_x = torch.cos(angles_x / 2), torch.sin(angles_x / 2)
        random_degrees_y = torch.rand(n, device=self.device) * 10 - 5 + 90
        angles_y = np.pi / 2 - random_degrees_y * (np.pi / 180)
        cos_half_y, sin_half_y = torch.cos(angles_y / 2), torch.sin(angles_y / 2)
        zeros = torch.zeros_like(sin_half_x)
        quat_x = torch.stack([sin_half_x, zeros, zeros, cos_half_x], dim=1)
        quat_y = torch.stack([zeros, sin_half_y, zeros, cos_half_y], dim=1)
        hand_quats = quat_mul(quat_y, quat_x)
        self._update_priv_buf(env_id=env_ids, name='hand_orientation', value=hand_quats)

        hand_pos = torch.zeros((n, 3), device=self.device, dtype=torch.float)
        hand_pos[:, 2] = 0.21
        hand_pos = hand_pos + torch.rand_like(hand_pos) * 0.001
        self._update_priv_buf(env_id=env_ids, name='hand_position', value=hand_pos)

        if self.object_tilt_enabled:
            tilt_angle = torch.rand(n, device=self.device) * 5.0 * (np.pi / 180)
            tilt_direction = torch.rand(n, device=self.device) * 2 * np.pi
            x_rotation = tilt_angle * torch.sin(tilt_direction)
            y_rotation = tilt_angle * torch.cos(tilt_direction)
            cos_half_x, sin_half_x = torch.cos(x_rotation / 2), torch.sin(x_rotation / 2)
            cos_half_y, sin_half_y = torch.cos(y_rotation / 2), torch.sin(y_rotation / 2)
            quat_x_obj = torch.stack([sin_half_x, zeros, zeros, cos_half_x], dim=1)
            quat_y_obj = torch.stack([zeros, sin_half_y, zeros, cos_half_y], dim=1)
            object_quats = quat_mul(quat_y_obj, quat_x_obj)
            object_root[:, 3:7] = object_quats
            self._update_priv_buf(env_id=env_ids, name='obj_orientation', value=object_quats)
            shaft_radius = 0.00625
            object_root[:, 2] += shaft_radius * torch.tan(tilt_angle)
        else:
            object_root[:, 3:7] = 0.0
            object_root[:, 6] = 1.0
            self._update_priv_buf(env_id=env_ids, name='obj_orientation', value=object_root[:, 3:7])

        # ---- write to sim (world frame, wxyz)
        origins = self.env_origins[env_ids]
        self.hand.write_root_pose_to_sim(torch.cat([hand_pos + origins, xyzw_to_wxyz(hand_quats)], dim=-1), env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device), env_ids=env_ids)
        self.object.write_root_pose_to_sim(torch.cat([object_root[:, :3] + origins, xyzw_to_wxyz(object_root[:, 3:7])], dim=-1), env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device), env_ids=env_ids)
        self.hand.write_joint_state_to_sim(hand_dof_pos, torch.zeros_like(hand_dof_pos), joint_ids=self.hand_dof_ids, env_ids=env_ids)
        # Deviation from IsaacGym: the original only reset the hand DOF state, so the screw joint angle kept
        # accumulating across episodes. Resetting it to 0 avoids hitting the 628 rad joint limit.
        screw_zero = torch.zeros((n, 1), device=self.device)
        self.object.write_joint_state_to_sim(screw_zero, screw_zero, env_ids=env_ids)
        self.object.instantaneous_wrench_composer.reset(env_ids)
        # keep the cached joint state consistent with what was written (IsaacGym wrote into the same view)
        self.xhand_hand_dof_pos[env_ids] = hand_dof_pos
        self.xhand_hand_dof_vel[env_ids] = 0

        self.contact_thresh[env_ids] = 0.05
        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.dof_vel_finite_diff[:] = 0
        self.nut_dof_pos_history[env_ids] = 0
        self.nut_contact_history[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    # ------------------------------------------------------------------
    # observations / rewards (verbatim logic from xhand_hora.py)
    # ------------------------------------------------------------------
    def compute_observations(self):
        self._refresh_gym()
        random_obs_noise_t = torch.normal(0, self.random_obs_noise_t_scale, size=self.xhand_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noisy_joint_pos = random_obs_noise_t + self.random_obs_noise_e + self.xhand_hand_dof_pos

        t_buf = (self.obs_buf_lag_history[:, -3:, :self.obs_buf.shape[1] // 3].reshape(self.num_envs, -1)).clone()
        self.obs_buf[:, :t_buf.shape[1]] = t_buf

        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None, :self.num_actions]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:self.numActions] = self.init_pose_buf[at_reset_env_ids, :self.num_actions].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, self.numActions:self.numActions * 2] = self.init_pose_buf[at_reset_env_ids, :self.num_actions].unsqueeze(1)

        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]
        self.nut_dof_vel_cf[at_reset_env_ids] = self.nut_dof_vel[at_reset_env_ids]

        self.at_reset_buf[at_reset_env_ids] = 0

        prev_nut_dof_pos_history = self.nut_dof_pos_history[:, 1:].clone()
        cur_nut_dof_pos = self.nut_dof_pos.clone().unsqueeze(1)
        self.nut_dof_pos_history[:] = torch.cat([prev_nut_dof_pos_history, cur_nut_dof_pos], dim=1)

        prev_nut_contact_history = self.nut_contact_history[:, 1:].clone()
        cur_nut_contact = self.nut_contact.clone().unsqueeze(1)
        self.nut_contact_history[:] = torch.cat([prev_nut_contact_history, cur_nut_contact], dim=1)

        if len(at_reset_env_ids) > 0:
            self.nut_dof_pos_history[at_reset_env_ids] = self.nut_dof_pos[at_reset_env_ids].unsqueeze(1).repeat(1, self.nut_termination_history_len, 1)
            self.nut_contact_history[at_reset_env_ids] = self.nut_contact[at_reset_env_ids].unsqueeze(1).repeat(1, self.nut_termination_history_len, 1)

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :self.numActions * 2]
        all_ids = range(self.num_envs)
        self._update_priv_buf(env_id=all_ids, name='obj_position', value=self.object_pos.clone())
        self._update_priv_buf(env_id=all_ids, name='obj_orientation', value=self.object_rot.clone())
        self._update_priv_buf(env_id=all_ids, name='obj_linvel', value=self.obj_linvel_at_cf.clone())
        self._update_priv_buf(env_id=all_ids, name='fingertip_orientation', value=self.fingertip_orientation.clone())
        self._update_priv_buf(env_id=all_ids, name='fingertip_linvel', value=self.ft_linvel_at_cf.clone())
        self._update_priv_buf(env_id=all_ids, name='fingertip_angvel', value=self.ft_angvel_at_cf.clone())
        self._update_priv_buf(env_id=all_ids, name='nut_pos', value=self.nut_pos.clone())
        self._update_priv_buf(env_id=all_ids, name='nut_dof_pos', value=self.nut_dof_pos.clone())
        self._update_priv_buf(env_id=all_ids, name='nut_dof_vel', value=self.nut_dof_vel_cf.clone())
        self._update_priv_buf(env_id=all_ids, name='fingertip_position', value=self.fingertip_pos.clone())

        if self.point_cloud_sampled_dim > 0:
            self.point_cloud_buf[:, :self.point_cloud_sampled_dim] = quat_apply(
                self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.obj_point_clouds
            ) + self.object_pos[:, None]

    def _agent_steps(self):
        env_steps = self._sim_frame_count * self.num_envs
        return env_steps // self.control_freq_inv

    def _get_reward_scale_by_name(self, name):
        agent_steps = self._agent_steps()
        init_scale, final_scale, curr_start, curr_end = self.reward_scale_dict[name]
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
        if self.evaluate:
            curr_progress = 1
        return init_scale + (final_scale - init_scale) * curr_progress

    def _get_current_angvel_penalty_threshold(self):
        agent_steps = self._agent_steps()
        init_threshold, final_threshold, curr_start, curr_end = self.angvel_penalty_threshold_curriculum
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
        if self.evaluate:
            curr_progress = 1
        return init_threshold + (final_threshold - init_threshold) * curr_progress

    def compute_reward(self, actions):
        current_angvel_penalty_threshold = self._get_current_angvel_penalty_threshold()

        pose_diff_penalty = (
            ((self.xhand_hand_dof_pos - self.init_pose_buf[..., :self.num_actions]) ** 2)
            * self.pose_diff_penalty_mask
        ).sum(-1)
        torque_penalty = (self.torques[:, -1, :self.num_actions] ** 2).sum(-1)
        work_penalty = (((torch.abs(self.torques[:, -1, :self.num_actions]) * torch.abs(self.dof_vel_finite_diff[:, -1])).sum(-1)) ** 2)

        angdiff = self.quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)

        nut_pos = self.nut_pos
        thumb_pos, index_pos = self._thumb_index_pos()
        thumb_dist, index_dist = torch.norm(thumb_pos - nut_pos, dim=-1), torch.norm(index_pos - nut_pos, dim=-1)
        mean_dist = 0.5 * (thumb_dist + index_dist)
        ratio = mean_dist / self.reset_dist_threshold
        proximity_reward = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

        nut_dof_linvel = ((self.nut_dof_pos.squeeze(-1) - self.nut_dof_pos_prev.squeeze(-1)) / (self.control_freq_inv * self.dt)).squeeze(-1)
        self.nut_dof_vel_cf = nut_dof_linvel.unsqueeze(-1)
        rotate_reward = torch.clip(nut_dof_linvel, max=self.angvel_clip_max, min=self.angvel_clip_min)
        rotate_penalty = torch.where(nut_dof_linvel > current_angvel_penalty_threshold, nut_dof_linvel - current_angvel_penalty_threshold, 0)

        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        self.obj_angvel_at_cf = object_angvel
        self.obj_linvel_at_cf = object_linvel
        ft_angdiff = self.quat_to_axis_angle(quat_mul(self.fingertip_orientation.reshape(-1, 4), quat_conjugate(self.ft_rot_prev.reshape(-1, 4)))).reshape(-1, self.fingers_num * 3)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt)
        self.ft_linvel_at_cf = ((self.fingertip_pos - self.ft_pos_prev) / (self.control_freq_inv * self.dt))
        self.z_dist_penalty = (self.object_pos[:, 2] - self.object_z) ** 2

        if self.point_cloud_sampled_dim > 0:
            point_cloud_z = self.point_cloud_buf[:, :self.point_cloud_sampled_dim, -1]
            z_dist_penalty = point_cloud_z.max(axis=1)[0] - point_cloud_z.min(axis=1)[0]
            z_dist_penalty[z_dist_penalty <= 0.03] = 0
        else:
            z_dist_penalty = to_torch([0], device=self.device)

        self.rew_buf[:] = compute_hand_reward(
            rotate_reward, self._get_reward_scale_by_name('rotate_reward'),
            pose_diff_penalty, self._get_reward_scale_by_name('pose_diff_penalty'),
            torque_penalty, self._get_reward_scale_by_name('torque_penalty'),
            work_penalty, self._get_reward_scale_by_name('work_penalty'),
            z_dist_penalty, self._get_reward_scale_by_name('pc_z_dist_penalty'),
            rotate_penalty, self._get_reward_scale_by_name('rotate_penalty'),
            proximity_reward, self._get_reward_scale_by_name('proximity_reward')
        )

        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['step_all_reward'] = self.rew_buf.mean()
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = torch.abs(object_angvel[:, 0]).mean()
        self.extras['pitch'] = torch.abs(object_angvel[:, 1]).mean()
        self.extras['yaw'] = torch.abs(object_angvel[:, 2]).mean()
        self.extras['z_dist_penalty'] = z_dist_penalty.mean()
        self.extras['rotate_penalty'] = rotate_penalty.mean()
        self.extras['proximity_reward'] = proximity_reward.mean()
        self.extras['curriculum/angvel_penalty_threshold'] = current_angvel_penalty_threshold
        self.extras['screw/angular_velocity'] = self.nut_dof_vel.mean()
        self.extras['screw/angular_position'] = self.nut_dof_pos.mean()
        self.extras['screw/positive_vel_ratio'] = (self.nut_dof_vel > 0).float().mean()
        self.extras['screw/nut_contact'] = (self.nut_contact > 1e-3).float().mean()

        if self.evaluate:
            vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
            for i in range(len(self.object_type_list)):
                env_ids = torch.where(self.object_type_at_env == i)
                if len(env_ids[0]) > 0:
                    running_mask = 1 - self.eval_done_buf[env_ids]
                    self.stat_sum_rewards[i] += (running_mask * self.rew_buf[env_ids]).sum()
                    self.stat_sum_episode_length[i] += running_mask.sum()
                    self.stat_sum_rotate_rewards[i] += (running_mask * rotate_reward[env_ids]).sum()
                    self.stat_sum_unclip_rotate_rewards[i] += (running_mask * vec_dot[env_ids]).sum()
                    if self.config['env']['object']['evalObjectType'] is not None:
                        flip = running_mask * self.reset_buf[env_ids]
                        self.env_evaluated[i] += flip.sum()
                        self.eval_done_buf[env_ids] += flip
                    tprint(f'Progress: {self.evaluate_iter} / {self.max_episode_length}')
            self.evaluate_iter += 1

    def check_termination(self, object_pos):
        term_by_max_eps = torch.greater_equal(self.progress_buf, self.max_episode_length)
        resets = term_by_max_eps

        nut_pos = self.nut_pos
        thumb_pos, index_pos = self._thumb_index_pos()
        thumb_dist = torch.norm(thumb_pos - nut_pos, dim=-1)
        index_dist = torch.norm(index_pos - nut_pos, dim=-1)
        finger_dist_reset = torch.logical_or(thumb_dist > self.reset_dist_threshold, index_dist > self.reset_dist_threshold)
        resets = torch.logical_or(resets, finger_dist_reset)

        nut_pos_history_filled = (self.progress_buf >= self.nut_termination_history_len)
        nut_pos_variance = torch.var(self.nut_dof_pos_history, dim=1).squeeze(-1)
        nut_pos_stagnant = (nut_pos_variance < self.nut_stagnation_eps) & nut_pos_history_filled
        resets = torch.logical_or(resets, nut_pos_stagnant)

        contact_history_filled = (self.progress_buf >= self.nut_termination_history_len)
        no_contact = torch.all(self.nut_contact_history <= 1e-3, dim=1).squeeze(-1)
        no_contact_reset = no_contact & contact_history_filled
        resets = torch.logical_or(resets, no_contact_reset)

        current_screw_pos = self.nut_dof_pos[:, 0]
        screw_at_limit = current_screw_pos > (self.screw_upper_limit - 5.0)
        resets = torch.logical_or(resets, screw_at_limit)
        return resets

    # ------------------------------------------------------------------
    # config parsing (verbatim from xhand_hora.py)
    # ------------------------------------------------------------------
    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.randomize_hand_scale = rand_config['randomize_hand_scale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.random_obs_noise_e_scale = rand_config['obs_noise_e_scale']
        self.random_obs_noise_t_scale = rand_config['obs_noise_t_scale']
        self.random_pose_noise = rand_config['pose_noise_scale']
        self.random_action_noise_e_scale = rand_config['action_noise_e_scale']
        self.random_action_noise_t_scale = rand_config['action_noise_t_scale']
        self.noisy_rpy_scale = rand_config['noisy_rpy_scale']
        self.noisy_pos_scale = rand_config['noisy_pos_scale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']
        self.contact_input_dim = p_config['contact_input_dim']
        self.contact_form = p_config['contact_form']
        self.contact_input = p_config['contact_input']
        self.contact_binarize_threshold = p_config['contact_binarize_threshold']
        self.enable_priv_obj_orientation = p_config['enable_obj_orientation']
        self.enable_priv_obj_linvel = p_config['enable_obj_linvel']
        self.enable_priv_obj_angvel = p_config['enable_obj_angvel']
        self.enable_priv_fingertip_position = p_config['enable_ft_pos']
        self.enable_priv_fingertip_orientation = p_config['enable_ft_orientation']
        self.enable_priv_fingertip_linvel = p_config['enable_ft_linvel']
        self.enable_priv_fingertip_angvel = p_config['enable_ft_angvel']
        self.enable_priv_hand_scale = p_config['enable_hand_scale']
        self.enable_priv_obj_restitution = p_config['enable_obj_restitution']
        self.enable_priv_tactile = p_config['enable_tactile']
        self.enable_priv_nut_contact = p_config['enable_nut_contact']
        self.enable_priv_nut_pos = p_config['enable_nut_pos']
        self.enable_priv_nut_dof_vel = p_config['enable_nut_dof_vel']
        self.enable_priv_nut_dof_pos = p_config['enable_nut_dof_pos']
        self.enable_priv_hand_position = p_config['enable_hand_position']
        self.enable_priv_hand_orientation = p_config['enable_hand_orientation']
        self.enable_priv_hand_joint_pos = p_config['enable_hand_joint_pos']
        self.enable_priv_pgain = p_config['enable_pgain']
        self.enable_priv_dgain = p_config['enable_dgain']
        self.enable_priv_screw_joint_friction = p_config.get('enable_screw_joint_friction', False)
        self.num_contacts = 0

        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        start_index = 0  # (sic) same as the original: overlaps the first 9 dims
        priv_dims = OrderedDict()
        priv_dims['obj_orientation'] = 4
        priv_dims['obj_linvel'] = 3
        priv_dims['obj_angvel'] = 3
        priv_dims['fingertip_position'] = 3 * self.fingers_num
        priv_dims['fingertip_orientation'] = 4 * self.fingers_num
        priv_dims['fingertip_linvel'] = self.fingers_num * 3
        priv_dims['fingertip_angvel'] = self.fingers_num * 3
        priv_dims['hand_scale'] = 1
        priv_dims['obj_restitution'] = 1
        priv_dims['tactile'] = self.num_contacts
        priv_dims['nut_contact'] = 1
        priv_dims['nut_pos'] = 3
        priv_dims['nut_dof_vel'] = 1
        priv_dims['nut_dof_pos'] = 1
        priv_dims['pgain'] = self.numActions
        priv_dims['dgain'] = self.numActions
        priv_dims['hand_joint_pos'] = self.numActions
        priv_dims['hand_orientation'] = 4
        priv_dims['hand_position'] = 3
        priv_dims['screw_joint_friction'] = 1
        for name, dim in priv_dims.items():
            if getattr(self, f'enable_priv_{name}'):
                self.priv_info_dict[name] = (start_index, start_index + dim)
                start_index += dim

    def _update_priv_buf(self, env_id, name, value):
        if getattr(self, f'enable_priv_{name}'):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)
        self.object_tilt_enabled = o_config.get('object_tilt', True)
        primitive_list = self.object_type.split('+')
        self.object_type_prob = []
        self.object_type_list = []
        for p_id, prim in enumerate(primitive_list):
            assert 'screw' in prim, 'only screw objects are ported'
            subset_name = self.object_type.split('_')[-1] if '_' in self.object_type else 'None'
            screw_compounds = sorted(glob(os.path.join(REPO_ROOT, f'assets/screw/{subset_name}/*.urdf')))
            screw_compound_list = [f'screw_{i}' for i in range(len(screw_compounds))]
            self.object_type_list += screw_compound_list
            self.object_type_prob += [raw_prob[p_id] / len(screw_compound_list) for _ in screw_compound_list]
        print(f'using {len(self.object_type_list)} training objects')

    def _allocate_task_buffer(self, num_envs):
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.priv_info_dim = max([v[1] for k, v in self.priv_info_dict.items()])
        self.point_cloud_sampled_dim = self.config['env']['hora']['point_cloud_sampled_dim']
        self.point_cloud_buffer_dim = self.point_cloud_sampled_dim
        self.priv_info_buf = torch.zeros((num_envs, self.priv_info_dim), device=self.device, dtype=torch.float)
        self.point_cloud_buf = torch.zeros((num_envs, self.point_cloud_sampled_dim, 3), device=self.device, dtype=torch.float)
        self.random_obs_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        self.random_action_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, self.numActions * 2), device=self.device, dtype=torch.float)
        self.nut_dof_pos_history = torch.zeros((num_envs, self.nut_termination_history_len, 1), device=self.device, dtype=torch.float)
        self.nut_contact_history = torch.zeros((num_envs, self.nut_termination_history_len, 1), device=self.device, dtype=torch.float)
        self.last_nut_pos = torch.zeros((num_envs,), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        self.reward_scale_dict = {}
        for k, v in r_config.items():
            if 'scale' in k:
                if type(v) is not omegaconf.listconfig.ListConfig and type(v) is not list:
                    v = [v, v, 0, 0]
                else:
                    assert len(v) == 4
                self.reward_scale_dict[k.replace('_scale', '')] = v
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        angvel_threshold_config = r_config.get('angvelPenaltyThres', 10.0)
        if type(angvel_threshold_config) is not omegaconf.listconfig.ListConfig and type(angvel_threshold_config) is not list:
            self.angvel_penalty_threshold_curriculum = [angvel_threshold_config, angvel_threshold_config, 0, 0]
        else:
            assert len(angvel_threshold_config) == 4
            self.angvel_penalty_threshold_curriculum = angvel_threshold_config
        self.angvel_penalty_threshold = self.angvel_penalty_threshold_curriculum[0]

    @staticmethod
    def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., 3:])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        return quaternions[..., :3] / sin_half_angles_over_angles

    # DirectRLEnv abstract hooks (unused: step() is overridden)
    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {}

    def _get_rewards(self):
        return self.rew_buf

    def _get_dones(self):
        return self.reset_buf.bool(), self.timeout_buf.bool()


def compute_hand_reward(
    rotate_reward, rotate_reward_scale: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
    z_dist_penalty, z_dist_penalty_scale: float,
    rotate_penalty, rotate_penalty_scale: float,
    proximity_reward, proximity_reward_scale: float
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    reward = reward + z_dist_penalty * z_dist_penalty_scale
    reward = reward + rotate_penalty * rotate_penalty_scale
    reward = reward + proximity_reward * proximity_reward_scale
    return reward
