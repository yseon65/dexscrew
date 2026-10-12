"""Measure finger jitter of dexscrew nut-bolt teacher / student rollouts (headless).

Fingertip second difference per control step (mm), split by fingertip distance to the nut (near = lowest 40 %,
far = highest 30 %, i.e. mostly out of contact), plus action saturation / sign-flip rates and reward-term means.

  ~/env_isaaclab6/bin/python scripts/measure_finger_jitter.py --policy teacher --preset teacher \
      [--ckpt outputs/.../best.pth] [hydra overrides ...]
"""
import argparse, os, sys, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--policy', choices=['teacher', 'student'], required=True)
p.add_argument('--preset', choices=['teacher', 'student'], required=True)
p.add_argument('--steps', type=int, default=600)
p.add_argument('--num_envs', type=int, default=256)
p.add_argument('--ckpt', default=None)
p.add_argument('--hold_steps', type=int, default=0, help='zero action for the first N steps of every episode (diagnostic)')
args, overrides = p.parse_known_args()
app = AppLauncher(headless=True).app
import torch, numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
DS = os.path.expanduser('~/IsaacLab/projects/dexscrew'); sys.path.insert(0, DS)
sys.path.insert(0, os.path.expanduser('~/IsaacLab/projects/dexdata'))
from dexdata import schema
from dexdata.adapters import DexscrewAdapter
from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.algo.ppo.ppo import PPO
from dexscrew.tasks import isaaclab_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
for name, fn in (('eq', lambda x, y: x.lower() == y.lower()), ('contains', lambda x, y: x.lower() in y.lower()),
                 ('if', lambda pred, a, b: a if pred else b), ('resolve_default', lambda d, a: d if a == '' else a)):
    OmegaConf.register_new_resolver(name, fn)
pre = {'student': ['task.env.reset_dist_threshold=0.12', 'task.env.randomization.obs_noise_t_scale=0.01',
                   'task.env.randomization.obs_noise_e_scale=0.02'],
       'teacher': ['task.env.reset_dist_threshold=0.07']}[args.preset]
with initialize_config_dir(config_dir=os.path.join(DS, 'configs'), version_base=None):
    config = compose('config', overrides=['task=XHandHoraNutBolt', f'task.env.numEnvs={args.num_envs}', 'test=True', 'seed=0',
        'task.env.object.type=screw_trinut', 'task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]',
        '+task.env.keepNutHeight=True'] + pre + overrides +
        (['train.algo=ProprioAdapt', 'train.ppo.proprio_adapt=True'] if args.policy == 'student' else ['train.algo=PPO']))
env = isaaclab_task_map['XHandHoraNutBolt'](omegaconf_to_dict(config.task), config.sim_device, 0, True)
O = os.path.join(DS, 'outputs')
agent = (ProprioAdapt if args.policy == 'student' else PPO)(env, os.path.join(O, 'watch'), full_config=config)
agent.restore_test(os.path.abspath(args.ckpt) if args.ckpt else os.path.join(O, 'XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt')
                   if args.policy == 'student' else os.path.join(O, 'XHandHoraNutBolt_teacher/lab_scale07125/stage1_nn/best_reward_1413.08.pth'))
agent.set_eval()
def pol(obs):
    pc = obs['point_cloud_info']
    if agent.normalize_point_cloud:
        pc = agent.point_cloud_mean_std(pc.reshape(-1, 3)).reshape((pc.shape[0], -1, 3))
    if args.policy == 'student':
        inp = {'obs': agent.running_mean_std(obs['obs']), 'proprio_hist': agent.sa_mean_std(obs['proprio_hist']), 'point_cloud_info': pc}
    else:
        inp = {'obs': agent.running_mean_std(obs['obs']),
               'priv_info': agent.priv_mean_std(obs['priv_info']) if agent.normalize_priv else obs['priv_info'],
               'proprio_hist': obs['proprio_hist'], 'point_cloud_info': pc}
    return agent.model.act_inference(inp)[0].clamp(-1, 1)
ad = DexscrewAdapter(env)
obs = env.reset()
cur_age = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
A, Q, KP, OP, D, NV = [], [], [], [], [], []
EX = {}
with torch.no_grad():
    for t in range(args.steps):
        st = ad.read_state()
        a = pol(obs)
        if args.hold_steps:
            a = torch.where((cur_age < args.hold_steps)[:, None], torch.zeros_like(a), a)
        obs, post, done, _ = ad.step(a)
        cur_age = torch.where(done.bool(), torch.zeros_like(cur_age), cur_age + 1)
        for k in ('rotation_reward', 'torques', 'work_done', 'joint_acc_penalty', 'action_rate_penalty', 'step_all_reward'):
            if k in env.extras:
                EX.setdefault(k, []).append(float(env.extras[k]))
        A.append(a.cpu()); Q.append(st['joint_pos'].cpu()); KP.append(st['keypoint_pos_wrist'][:, schema.KEYPOINT_SETS['tips']].cpu())
        OP.append(st['object_pos_wrist'].cpu()); D.append(done.cpu()); NV.append(st['object_dof_vel'][:, 0].cpu())
A, Q, KP, OP, D, NV = (torch.stack(x).numpy() for x in (A, Q, KP, OP, D, NV))
# valid triples (t-1, t, t+1) without reset in between
ok = ~(D[:-2] | D[1:-1])
J = {'index': [0, 1, 2], 'thumb': [9, 10, 11], 'mid': [3, 4]}
TIP = {'index': 0, 'mid': 1, 'thumb': 4}
acc = KP[2:] - 2 * KP[1:-1] + KP[:-2]                          # (T-2,N,5,3) fingertip 2nd diff
res = {'policy': args.policy, 'preset': args.preset, 'ckpt': args.ckpt, 'overrides': overrides, 'nut_rad_s': float(NV.mean()),
       'reward_terms': {k: float(np.mean(v)) for k, v in EX.items()}}
for f, js in J.items():
    a = A[1:-1][..., js]; a0 = A[:-2][..., js]
    flip = (np.sign(a) != np.sign(a0)) & (np.abs(a) > 0.5) & (np.abs(a0) > 0.5)
    tipacc = np.linalg.norm(acc[:, :, TIP[f]], axis=-1) * 1e3     # mm / step^2
    dist = np.linalg.norm(KP[1:-1, :, TIP[f]] - OP[1:-1], axis=-1)
    thr_near, thr_far = np.quantile(dist[ok], [0.4, 0.7])
    res[f] = {'sat': float((np.abs(a) > 0.99)[ok].mean()), 'flip': float(flip[ok].mean()),
              'tip_acc_mm': float(tipacc[ok].mean()),
              'tip_acc_near_mm': float(tipacc[ok & (dist < thr_near)].mean()),
              'tip_acc_far_mm': float(tipacc[ok & (dist > thr_far)].mean()),
              'flip_far': float(flip[ok & (dist > thr_far)].mean())}
# acceleration vs. steps since episode start (age of the centre step of each triple)
age = np.zeros(D.shape, dtype=int)
for t in range(1, len(D)):
    age[t] = np.where(D[t - 1], 0, age[t - 1] + 1)
age_c = age[1:-1]
jacc = np.abs(Q[2:] - 2 * Q[1:-1] + Q[:-2]).sum(-1)             # (T-2,N) joint 2nd diff, rad / step^2
tip_all = np.linalg.norm(acc, axis=-1).mean(-1) * 1e3           # (T-2,N) mean over 5 fingertips, mm
res['by_age'] = {}
for lo, hi in ((1, 3), (3, 6), (6, 11), (11, 21), (21, 51), (51, 10 ** 9)):
    m = ok & (age_c >= lo) & (age_c < hi)
    if m.any():
        res['by_age'][f'{lo}-{hi - 1 if hi < 10 ** 9 else "inf"}'] = {
            'n': int(m.sum()), 'joint_acc_rad': float(jacc[m].mean()), 'tip_acc_mm': float(tip_all[m].mean()),
            'thumb_tip_acc_mm': float((np.linalg.norm(acc[:, :, TIP['thumb']], axis=-1) * 1e3)[m].mean()),
            'sat': float((np.abs(A[1:-1]) > 0.99)[m].mean())}
print('[jitter] ' + json.dumps(res), flush=True)
os._exit(0)
