"""Deterministic evaluation of teacher checkpoints (headless).

For each checkpoint, rolls out the mean action in N envs for a fixed number of steps and reports
episode reward / length, timeout ratio, nut angular velocity, total rotation per episode and the
breakdown of termination causes.

Usage:
  ~/env_isaaclab6/bin/python scripts/eval_policy.py --task XHandHoraNutBolt --num_envs 256 --steps 1600 \
      task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.07 \
      --ckpts outputs/XHandHoraNutBolt_teacher/lab_run1/stage1_nn/last.pth ...
  (hydra overrides must come before --ckpts)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='XHandHoraNutBolt')
parser.add_argument('--ckpts', nargs='+', required=True)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--steps', type=int, default=1600)
parser.add_argument('--student', action='store_true',
                    help='checkpoints are ProprioAdapt students (.ckpt): act from proprio history, no priv info')
args, overrides = parser.parse_known_args()
app = AppLauncher(headless=True).app

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
from dexscrew.algo.ppo.padapt import ProprioAdapt  # noqa: E402
from dexscrew.algo.ppo.ppo import PPO  # noqa: E402
from dexscrew.tasks import isaaclab_task_map  # noqa: E402
from dexscrew.utils.reformat import omegaconf_to_dict  # noqa: E402

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

with initialize_config_dir(config_dir=os.path.join(REPO, 'configs'), version_base=None):
    config = compose('config', overrides=[f'task={args.task}', f'task.env.numEnvs={args.num_envs}',
                                          'test=True'] + overrides
                     + (['train.algo=ProprioAdapt', 'train.ppo.proprio_adapt=True'] if args.student else ['train.algo=PPO']))
env = isaaclab_task_map[args.task](omegaconf_to_dict(config.task), config.sim_device, 0, True)
agent_cls = ProprioAdapt if args.student else PPO
agent = agent_cls(env, os.path.join(REPO, 'outputs', 'watch'), full_config=config)
agent.set_eval()

# record which termination condition fired (same logic as XHandHoraLab.check_termination)
cause = {}
_orig_check = env.check_termination


def _check(object_pos):
    th, ix = env._thumb_index_pos()
    far = (torch.norm(th - env.nut_pos, dim=-1) > env.reset_dist_threshold) | \
          (torch.norm(ix - env.nut_pos, dim=-1) > env.reset_dist_threshold)
    filled = env.progress_buf >= env.nut_termination_history_len
    stagnant = (torch.var(env.nut_dof_pos_history, dim=1).squeeze(-1) < env.nut_stagnation_eps) & filled
    no_contact = torch.all(env.nut_contact_history <= 1e-3, dim=1).squeeze(-1) & filled
    timeout = env.progress_buf >= env.max_episode_length
    cause.update(far=far, stagnant=stagnant, no_contact=no_contact, timeout=timeout,
                 limit=env.nut_dof_pos[:, 0] > env.screw_upper_limit - 5.0)
    return _orig_check(object_pos)


env.check_termination = _check


def run(ckpt):
    agent.restore_test(ckpt)
    env.reset_buf[:] = 1
    obs = env.reset()
    n = args.num_envs
    ep_rew = torch.zeros(n, device=env.device)
    ep_len = torch.zeros(n, device=env.device)
    ep_rot = torch.zeros(n, device=env.device)
    rews, lens, rots = [], [], []
    counts = dict(far=0, stagnant=0, no_contact=0, timeout=0, limit=0)
    vel_sum, vel_pos, vel_n = 0.0, 0.0, 0
    for _ in range(args.steps):
        pc = obs['point_cloud_info']
        if agent.normalize_point_cloud:
            pc = agent.point_cloud_mean_std(pc.reshape(-1, 3)).reshape((pc.shape[0], -1, 3))
        if args.student:  # same inputs as ProprioAdapt.test(): no privileged info
            inp = {'obs': agent.running_mean_std(obs['obs']),
                   'proprio_hist': agent.sa_mean_std(obs['proprio_hist']), 'point_cloud_info': pc}
        else:
            inp = {'obs': agent.running_mean_std(obs['obs']),
                   'priv_info': agent.priv_mean_std(obs['priv_info']) if agent.normalize_priv else obs['priv_info'],
                   'proprio_hist': obs['proprio_hist'], 'point_cloud_info': pc}
        mu, _, _ = agent.model.act_inference(inp)
        obs, r, done, _ = env.step(torch.clamp(mu, -1.0, 1.0))
        v = env.nut_dof_vel_cf.squeeze(-1)
        ep_rew += r
        ep_len += 1
        ep_rot += v * env.control_freq_inv * env.dt
        vel_sum += v.sum().item()
        vel_pos += (v > 0).float().sum().item()
        vel_n += n
        d = done.nonzero().squeeze(-1)
        if len(d):
            rews += ep_rew[d].tolist()
            lens += ep_len[d].tolist()
            rots += ep_rot[d].tolist()
            for k in counts:
                counts[k] += cause[k][d].sum().item()
            ep_rew[d] = 0
            ep_len[d] = 0
            ep_rot[d] = 0
    ne = max(len(lens), 1)
    print(f'[eval] {os.path.basename(ckpt)}: episodes {len(lens)} | reward {sum(rews)/ne:.1f} | '
          f'length {sum(lens)/ne:.0f} | timeout(800) {counts["timeout"]/ne:.1%} | '
          f'rotation/episode {sum(rots)/ne:.1f} rad | nut vel {vel_sum/vel_n:.2f} rad/s | '
          f'positive vel {vel_pos/vel_n:.1%} | causes (not exclusive): '
          f'far {counts["far"]/ne:.1%}, stagnant {counts["stagnant"]/ne:.1%}, '
          f'no_contact {counts["no_contact"]/ne:.1%}, joint_limit {counts["limit"]/ne:.1%}', flush=True)


for c in args.ckpts:
    run(c)
os._exit(0)
