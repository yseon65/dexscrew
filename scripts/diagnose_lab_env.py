"""Quick sanity check of the Isaac Lab port: runs zero / random actions and prints termination stats.

Usage: ~/env_isaaclab6/bin/python scripts/diagnose_lab_env.py [num_envs] [steps] [task] [hydra overrides...]
  e.g. scripts/diagnose_lab_env.py 16 120 XHandHoraNutBolt task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.07
"""
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import os  # noqa: E402

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dexscrew.tasks import isaaclab_task_map  # noqa: E402
from dexscrew.utils.reformat import omegaconf_to_dict  # noqa: E402

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
task = sys.argv[3] if len(sys.argv) > 3 else 'XHandHoraScrewDriver'
overrides = sys.argv[4:] if len(sys.argv) > 4 else ['task.env.reset_dist_threshold=0.1']
with initialize_config_dir(config_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')), version_base=None):
    cfg = compose('config', overrides=[f'task={task}', f'task.env.numEnvs={num_envs}'] + overrides)
task_cfg = omegaconf_to_dict(cfg.task)
env = isaaclab_task_map[task](task_cfg, 'cuda:0', 0, True)
print('object bodies', env.object.body_names, 'joints', env.object.joint_names, 'limits', env.object.data.joint_pos_limits[0].tolist())
_orig_check = env.check_termination
def _check(object_pos):
    th, ix = env._thumb_index_pos()
    td, idd = torch.norm(th - env.nut_pos, dim=-1), torch.norm(ix - env.nut_pos, dim=-1)
    print(f'  [term] progress={env.progress_buf[:4].tolist()} max_len={env.max_episode_length} thumb_d={td[:4].tolist()} '
          f'index_d={idd[:4].tolist()} nut_dof={env.nut_dof_pos[:4, 0].tolist()} hand_root={(env.hand.data.root_link_pos_w - env.env_origins)[0].tolist()}')
    print('    quat(wxyz)', env.hand.data.root_link_quat_w[0].tolist(), 'obj', (env.object.data.root_link_pos_w - env.env_origins)[0].tolist(), env.object.data.root_link_quat_w[0].tolist())
    print('    q', env.xhand_hand_dof_pos[0].tolist())
    print('    init', env.init_pose_buf[0, :12].tolist())
    return _orig_check(object_pos)
env.check_termination = _check
print('priv_info_dim', env.priv_info_dim, 'obs', env.observation_space.shape, 'act', env.action_space.shape)
obs = env.reset()
print({k: tuple(v.shape) for k, v in obs.items()})
thumb, index = env._thumb_index_pos()
print('nut_pos[0]', env.nut_pos[0].tolist())
print('thumb dist', torch.norm(thumb - env.nut_pos, dim=-1)[:8].tolist())
print('index dist', torch.norm(index - env.nut_pos, dim=-1)[:8].tolist())
print('fingertips env0', env.fingertip_pos[0].view(5, 3).tolist())
ep_len = torch.zeros(num_envs, device=env.device)
lens = []
for t in range(steps):
    a = torch.zeros((num_envs, 12), device=env.device) if t < steps // 2 else torch.rand((num_envs, 12), device=env.device) * 2 - 1
    prev_progress = env.progress_buf.clone()
    obs, r, done, info = env.step(a)
    if t % 10 == 0:
        th, ix = env._thumb_index_pos()
        print(f't={t} rew={r.mean():.3f} done={done.float().mean():.2f} contact={info["screw/nut_contact"]:.2f} '
              f'nut_vel={env.nut_dof_vel.mean():.3f} thumb_d={torch.norm(th - env.nut_pos, dim=-1).mean():.3f} '
              f'index_d={torch.norm(ix - env.nut_pos, dim=-1).mean():.3f} '
              f'rot={info["rotation_reward"]:.3f} prox={info["proximity_reward"]:.3f} pose={info["pose_diff_penalty"]:.3f} '
              f'torq={info["torques"]:.4f}')
    ep_len += 1
    d = done.nonzero().squeeze(-1)
    lens += ep_len[d].tolist()
    ep_len[d] = 0
print('finished episodes', len(lens), 'mean length', sum(lens) / max(len(lens), 1))
print('priv[0]', env.priv_info_buf[0].tolist())
os._exit(0)
