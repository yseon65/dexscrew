# train_lab.py
# Isaac Lab version of train.py. Same hydra configs / command line overrides, e.g.
#   python train_lab.py task=XHandHoraScrewDriver headless=True seed=42 train.algo=PPO ...
# The Isaac Sim app must be launched before any isaaclab / omni module is imported,
# so the `headless` and `sim_device` overrides are peeked from sys.argv first.

import sys

from isaaclab.app import AppLauncher


def _peek_override(key, default):
    for arg in sys.argv[1:]:
        if arg.startswith(f'{key}='):
            return arg.split('=', 1)[1]
    return default


_headless = _peek_override('headless', 'False').lower() == 'true'
_device = _peek_override('sim_device', 'cuda:0')
app_launcher = AppLauncher(headless=_headless, device=_device)
simulation_app = app_launcher.app

import datetime  # noqa: E402
import os  # noqa: E402

import hydra  # noqa: E402
from hydra.utils import to_absolute_path  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from termcolor import cprint  # noqa: E402

from dexscrew.algo.ppo.padapt import ProprioAdapt  # noqa: E402,F401
from dexscrew.algo.ppo.ppo import PPO  # noqa: E402,F401
from dexscrew.tasks import isaaclab_task_map  # noqa: E402
from dexscrew.utils.misc import git_diff_config, git_hash, set_np_formatting, set_seed  # noqa: E402
from dexscrew.utils.reformat import omegaconf_to_dict, print_dict  # noqa: E402

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name='config', config_path='configs', version_base=None)
def main(config: DictConfig):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cache_name = os.path.basename(config.train.ppo.output_name)
    run_name = f"{cache_name}_{time_str}" if cache_name else f"{config.wandb_name}_{time_str}"

    if config.checkpoint:
        if '*' in config.checkpoint:
            from glob import glob
            _ckpt = glob(config.checkpoint)
            assert len(_ckpt) == 1, _ckpt
            config.checkpoint = _ckpt[0]
        config.checkpoint = to_absolute_path(config.checkpoint)

    set_np_formatting()
    cfg_dict = omegaconf_to_dict(config)
    print_dict(cfg_dict)
    config.seed = set_seed(config.seed)

    if config.wandb_activate:
        import wandb
        wandb.init(project=config.wandb_project, config=cfg_dict, sync_tensorboard=True, name=run_name,
                   resume="allow")

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    task_config = omegaconf_to_dict(config.task)
    task_config['seed'] = config.seed
    env = isaaclab_task_map[config.task_name](
        config=task_config,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )

    output_dif = os.path.join('outputs', config.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(config.train.algo)(env, output_dif, full_config=config)
    if config.test:
        assert config.train.load_path
        agent.restore_test(config.train.load_path)
        agent.test()
    else:
        date = str(datetime.datetime.now().strftime('%m%d%H'))
        print(git_diff_config('./'))
        os.system(f'git diff HEAD > {output_dif}/gitdiff.patch')
        with open(os.path.join(output_dif, f'config_{date}_{git_hash()}.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(config))
        agent.restore_train(config.train.load_path)
        agent.train()


if __name__ == '__main__':
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # SimulationApp.close() can hang on shutdown; exit hard once done.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
