"""Live viewer for a running training job.

Opens an Isaac Sim GUI with a few envs, rolls out the newest checkpoint of a run deterministically,
and hot-reloads whenever a newer checkpoint appears (checked every --reload_sec seconds).
Runs alongside training (small GPU footprint).

Usage:
  ~/env_isaaclab6/bin/python scripts/watch_policy.py --task XHandHoraNutBolt \
      --run outputs/XHandHoraNutBolt_teacher/lab_run1 --num_envs 4 \
      task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.07
"""
import argparse
import glob
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='XHandHoraNutBolt')
parser.add_argument('--run', required=True, help='training output dir (contains stage1_nn/)')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--reload_sec', type=float, default=120.0)
parser.add_argument('--headless', action='store_true', help='no GUI, only print stats (for testing)')
args, overrides = parser.parse_known_args()
app = AppLauncher(headless=args.headless).app

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
from dexscrew.algo.ppo.ppo import PPO  # noqa: E402
from dexscrew.tasks import isaaclab_task_map  # noqa: E402
from dexscrew.utils.reformat import omegaconf_to_dict  # noqa: E402

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


def newest_checkpoint(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, 'stage1_nn', '*.pth'))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


with initialize_config_dir(config_dir=os.path.join(REPO, 'configs'), version_base=None):
    config = compose('config', overrides=[f'task={args.task}', f'task.env.numEnvs={args.num_envs}',
                                          'train.algo=PPO', 'test=True'] + overrides)
task_config = omegaconf_to_dict(config.task)
env = isaaclab_task_map[args.task](task_config, config.sim_device, 0, args.headless)
watch_dir = os.path.join(REPO, 'outputs', 'watch')
agent = PPO(env, watch_dir, full_config=config)
agent.set_eval()

ckpt, last_check = None, 0.0
obs_dict = env.reset()
ep_rew = torch.zeros(args.num_envs, device=env.device)
ep_len = torch.zeros(args.num_envs, device=env.device)
done_rews, done_lens = [], []
while app.is_running():
    if time.time() - last_check > args.reload_sec or ckpt is None:
        last_check = time.time()
        new = newest_checkpoint(os.path.join(REPO, args.run) if not os.path.isabs(args.run) else args.run)
        if new is None:
            print('[watch] no checkpoint yet, waiting ...', flush=True)
            time.sleep(10)
            continue
        if new != ckpt:
            agent.restore_test(new)
            ckpt = new
            print(f'[watch] loaded {os.path.basename(ckpt)}', flush=True)
        if done_lens:
            print(f'[watch] last episodes: reward {sum(done_rews) / len(done_rews):.1f}, '
                  f'length {sum(done_lens) / len(done_lens):.0f}, nut vel {env.nut_dof_vel.mean():.2f} rad/s', flush=True)
            done_rews, done_lens = [], []

    pc = obs_dict['point_cloud_info']
    if agent.normalize_point_cloud:
        pc = agent.point_cloud_mean_std(pc.reshape(-1, 3)).reshape((pc.shape[0], -1, 3))
    input_dict = {
        'obs': agent.running_mean_std(obs_dict['obs']),
        'priv_info': agent.priv_mean_std(obs_dict['priv_info']) if agent.normalize_priv else obs_dict['priv_info'],
        'proprio_hist': obs_dict['proprio_hist'],
        'point_cloud_info': pc,
    }
    mu, _, _ = agent.model.act_inference(input_dict)
    obs_dict, r, done, _ = env.step(torch.clamp(mu, -1.0, 1.0))
    ep_rew += r
    ep_len += 1
    d = done.nonzero().squeeze(-1)
    if len(d):
        done_rews += ep_rew[d].tolist()
        done_lens += ep_len[d].tolist()
        ep_rew[d] = 0
        ep_len[d] = 0

os._exit(0)
