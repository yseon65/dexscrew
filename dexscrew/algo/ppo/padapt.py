# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: In-Hand Object Rotation via Rapid Motor Adaptation
# Copyright (c) 2022 Haozhi Qi
# Licensed under MIT License
# https://github.com/HaozhiQi/hora/
# --------------------------------------------------------

import os
import time
import torch
import numpy as np
from termcolor import cprint

from dexscrew.utils.misc import AverageScalarMeter, tprint
from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


class ProprioAdapt(object):
    def __init__(self, env, output_dir, full_config, student_dim=24):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        self.proprio_dim = self.ppo_config.get('proprio_dim', 24)
        self.student_obs_shape = (student_dim * 3,)
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.normalize_priv = self.ppo_config['normalize_priv']
        self.priv_info_dim = self.env.priv_info_dim
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        self.proprio_hist_dim = self.env.prop_hist_len
        # ---- Critic Info
        self.asymm_actor_critic = self.ppo_config.get('asymm_actor_critic', False)
        self.critic_info_dim = self.ppo_config.get('critic_info_dim', 0)
        # ---- Point Cloud / Proprio Hist Info
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        self.proprio_len = self.ppo_config['proprio_len']
        self.use_point_cloud_info = self.ppo_config['use_point_cloud_info']
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'critic_info_dim': self.critic_info_dim,
            'asymm_actor_critic': self.asymm_actor_critic,
            'point_mlp_units': self.network_config.point_mlp.units,
            'use_point_cloud_info': self.use_point_cloud_info,
            'proprio_len': self.proprio_len,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, self.proprio_dim)).to(self.device)
        self.sa_mean_std.train()
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        self.point_cloud_mean_std.eval()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        writer = SummaryWriter(self.tb_dir)
        self.writer = writer
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        # ---- Optim ----
        adapt_params = []
        for name, p in self.model.named_parameters():
            if 'adapt_tconv' in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

    def recon_criterion(self, out, target):
        return (out - target).pow(2)

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        c = 0
        while True:
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(obs_dict['point_cloud_info'].reshape(-1, 3)).reshape((obs_dict['obs'].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict['point_cloud_info']
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
                'point_cloud_info': point_cloud_info,
            }
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            c += 1
            print(f"Step {c}")

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= 1e9:
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(obs_dict['point_cloud_info'].reshape(-1, 3)).reshape((obs_dict['obs'].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict['point_cloud_info']
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']).detach(),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info'],
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
                'point_cloud_info': point_cloud_info,
            }
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
            latent_loss = ((e - e_gt.detach()) ** 2).mean() # Latent loss (student-teacher)

            teacher_obs_input = torch.cat([input_dict['obs'], e_gt.detach()], dim=-1)
            with torch.no_grad():
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = self.model.mu(teacher_x)

            #  Behavior cloning loss between student action and teacher action
            bc_loss = torch.sum(self.recon_criterion(torch.clamp(mu, -1, 1),torch.clamp(teacher_mu, -1, 1)), dim=-1).mean()

            # Total loss: latent loss + behavior cloning loss
            loss = bc_loss + latent_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.direct_info['latent_loss'] = float(latent_loss.detach().cpu())
            self.direct_info['bc_loss'] = float(bc_loss.detach().cpu())
            self.direct_info['total_loss'] = float(loss.detach().cpu())

            self.log_tensorboard()

            # `agent_steps % 1e8 == 0` never holds when the batch size does not divide 1e8 (e.g. 48 envs),
            # so save whenever a 10M (model_last) / 100M (snapshot) boundary is crossed instead.
            prev_steps = self.agent_steps - self.batch_size
            if self.agent_steps // 1e8 > prev_steps // 1e8:
                self.save(os.path.join(self.nn_dir, f'{self.agent_steps // 1e8}00m'))
            if self.agent_steps // 1e7 > prev_steps // 1e7:
                self.save(os.path.join(self.nn_dir, f'model_last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, f'model_best'))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)

    def log_tensorboard(self):
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

    def restore_train(self, fn):
        checkpoint = torch.load(fn, weights_only=False)
        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'priv_mean_std' in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, weights_only=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights['sa_mean_std'] = self.sa_mean_std.state_dict()
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()
        torch.save(weights, f'{name}.ckpt')
