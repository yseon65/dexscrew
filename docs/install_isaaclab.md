# Running DexScrew on Isaac Lab (Isaac Sim 6.0)

The original code targets IsaacGym Preview 4 (Python 3.8). IsaacGym does not run on recent GPUs such as Blackwell (RTX 50xx), because PyTorch builds for Python 3.8 lack `sm_120` support. This branch therefore adds an Isaac Lab port of the simulation environment. The PPO / ProprioAdapt code, the networks and the hydra configs are unchanged.

Tested with:
- RTX 5080 16GB, driver 580
- Isaac Sim 6.0 + Isaac Lab (`isaaclab` 0.54.4)
- a uv venv with Python 3.12 and torch 2.11+cu128

## Setup

```bash
# extra python deps inside the Isaac Lab venv (no numpy downgrade needed)
uv pip install --python ~/env_isaaclab6/bin/python hydra-core termcolor tensorboardX tensorboard h5py

# one-time URDF -> USD conversion (writes assets/usd/)
~/env_isaaclab6/bin/python scripts/convert_urdf_to_usd.py
```

The conversion is done offline because Isaac Lab's `UrdfFileCfg` still calls the legacy `_urdf` interface, which Isaac Sim 6.0 no longer ships. The conversion script does three things:
- it fixes the screw URDF mesh paths (`../meshes` should point to `assets/meshes`);
- it sets the contact offset to 0.002 and the rest offset to 0.0;
- it disables the importer's world `root_joint`, which would otherwise ignore root-pose resets. The env uses `fix_root_link=True` instead.

## Training

```bash
# Step 1: oracle / teacher (8192 envs, about 11.5GB GPU memory, about 42k FPS on an RTX 5080)
scripts/lab_screwdriver_teacher.sh 0 42 run1 train.ppo.max_agent_steps=1000000000

# Step 2: student (ProprioAdapt), distilled from the teacher checkpoint
scripts/lab_screwdriver_student_padapt.sh 0 42 run1 \
    checkpoint=outputs/XHandHoraScrewDriver_teacher/run1/stage1_nn/last.pth
```

Nut-bolt task (default object `screw_trinut`; override with `task.env.object.type=screw_boxnut` or `screw_crossnut`):

```bash
scripts/lab_nutbolt_teacher.sh 0 42 run1
scripts/lab_nutbolt_student_padapt.sh 0 42 run1 \
    checkpoint=outputs/XHandHoraNutBolt_teacher/run1/stage1_nn/last.pth
```

All scripts accept the same arguments as their IsaacGym counterparts. `scripts/diagnose_lab_env.py` prints termination and contact stats for a quick sanity check.

Other tools:
- `scripts/eval_policy.py` evaluates checkpoints deterministically and breaks down why episodes ended.
- `scripts/watch_policy.py` is a GUI viewer that reloads the newest checkpoint of a running job.

Training results and a log of the porting work are in [isaaclab_reproduction_report.md](isaaclab_reproduction_report.md) (in Korean).

## Differences from the IsaacGym version

| | IsaacGym | Isaac Lab port |
|---|---|---|
| Object scale | per-env `U(s-0.025, s+0.025)` around bucket `s = list[i % 9]` | exactly `list[i % 9]`, because articulations cannot be rescaled at runtime (assigned with `MultiAssetSpawnerCfg(random_choice=False)`) |
| Screwdriver type | random per env (`0000` / `0001`) | deterministic per env (asset `i % 18`, balanced) |
| Screw joint on reset | not reset; the angle accumulated across episodes | reset to 0 |
| Hand effort limits | clamped by the DOF `effort` property | clamped explicitly in Python with the same per-DOF values |
| Nut joint limit | the nut URDFs have no `<limit>` | the importer sets [0, 628.3] rad (same as the screwdriver) |
| Physics | PhysX 4 (IsaacGym) | PhysX 5 (Isaac Sim 6.0); the solver settings, contact offsets and damping are copied, but the contact dynamics are not identical |

Everything else is ported 1:1: rewards, terminations, observation and privileged-info layout, curriculum, PD-gain, noise and random-force randomization, and friction, mass and COM randomization. Internally the env keeps IsaacGym's `xyzw` quaternions and env-local positions, so `priv_info` has the same layout.
