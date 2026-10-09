# DexScrew Isaac Lab 포팅 및 RL 학습 재현 기록

- 작업 기간: 2026-10-08 ~ 2026-10-09
- 브랜치: `isaaclab-port` (base: `3bde4e3`)
- 설치/실행 방법 요약: [install_isaaclab.md](install_isaaclab.md)

## 1. 배경

원본 레포는 **IsaacGym Preview 4 (Python 3.8)** 기반이다. 작업 PC의 환경은 다음과 같다.

| 항목 | 값 |
|---|---|
| GPU | RTX 5080 16GB (Blackwell, `sm_120`), driver 580 / CUDA 13.0 |
| 시뮬레이터 | Isaac Sim 6.0 + Isaac Lab main (`isaaclab` 0.54.4) |
| Python | uv venv `~/env_isaaclab6` (Python 3.12, torch 2.11+cu128). conda 없음 |

이 GPU에서는 IsaacGym을 실행할 수 없다. Python 3.8용 torch 빌드가 `sm_120`을 지원하지 않기 때문이다. 그래서 **시뮬레이션 환경(task)만 Isaac Lab으로 포팅**했다. 다음은 그대로 재사용했다.

- PPO와 ProprioAdapt 알고리즘 (`dexscrew/algo`)
- 네트워크
- hydra config

재현 범위는 README의 Step 1(oracle/teacher PPO)과 Step 2(student ProprioAdapt)이다. screwdriver와 nut-bolt 두 task 모두 포팅했다.

## 2. 코드 변경 사항

### 새로 추가한 파일

| 파일 | 내용 |
|---|---|
| `dexscrew/tasks/xhand_hora_lab.py` | `xhand_hora.py`의 Isaac Lab 포팅 (`DirectRLEnv` 기반, 원본 VecTask 인터페이스 유지) |
| `dexscrew/utils/torch_jit_utils.py` | IsaacGym `torch_utils`의 xyzw 쿼터니언 함수 이식 |
| `train_lab.py` | `train.py`의 Isaac Lab 버전. hydra 인자 동일, AppLauncher 선기동 |
| `scripts/convert_urdf_to_usd.py` | URDF → USD 1회 변환 (`assets/usd/<subset>/`에 생성) |
| `scripts/lab_screwdriver_{teacher,student_padapt}.sh` | 원본 스크립트와 같은 인자를 받는 Isaac Lab 실행 스크립트 |
| `scripts/lab_nutbolt_{teacher,student_padapt}.sh` | 위와 동일 (nut-bolt) |
| `scripts/diagnose_lab_env.py` | zero/random action으로 종료 원인·접촉·거리 진단 |
| `scripts/eval_policy.py` | 체크포인트 결정적 평가 (teacher / `--student`), 종료 원인 분해 |
| `scripts/watch_policy.py` | GUI viewer. 학습 중 최신 체크포인트를 주기적으로 다시 로드 |
| `docs/install_isaaclab.md` | 설치·실행 방법, 원본과의 차이 표 |

### 수정한 원본 파일 (최소 변경)

- `dexscrew/tasks/__init__.py`
  - task map을 lazy import로 바꿨다. isaacgym이 없어도 import할 수 있다.
  - `isaaclab_task_map`을 추가했다.
- `dexscrew/algo/ppo/ppo.py`, `padapt.py`: `torch.load(..., weights_only=False)`로 바꿨다. torch 2.6부터 `weights_only`의 기본값이 바뀌었기 때문이다.
- `dexscrew/algo/ppo/experience.py`: `gym` 대신 `gymnasium`을 쓴다.
- `dexscrew/algo/ppo/padapt.py`: **checkpoint 저장 버그를 고쳤다.**
  - 원래 조건은 `agent_steps % 1e8 == 0`이다. batch 크기(48 등)가 1e8을 나누지 못하면 이 조건이 한 번도 참이 되지 않는다.
  - 이제 10M step마다 `model_last`를, 100M step마다 snapshot을 저장한다.

### 추가로 설치한 패키지 (Isaac Lab venv)

```bash
uv pip install --python ~/env_isaaclab6/bin/python hydra-core termcolor tensorboardX tensorboard h5py
```

`requirements.txt`의 `numpy==1.22.4`는 무시했다. 기존 numpy 1.26을 그대로 쓴다.

## 3. 포팅 중 발견한 문제와 해결

1. **Isaac Lab의 `UrdfFileCfg`가 Isaac Sim 6.0에서 동작하지 않는다.**
   - 원인: 레거시 `isaacsim.asset.importer.urdf._urdf` 모듈이 없다.
   - 해결: 새 `URDFImporter`로 오프라인 변환한 뒤 `UsdFileCfg`로 로드한다.
2. **screw URDF의 mesh 경로가 틀렸다.**
   - URDF는 `../meshes/*.stl`을 가리키지만 실제 mesh는 `assets/meshes/`에 있다.
   - 그대로 변환하면 핸들(nut) collider가 비게 된다.
   - 해결: 변환 스크립트가 경로를 고친 임시 URDF를 만들어 변환한다.
3. **root pose reset이 무시됐다.**
   - 원인: importer가 world에 고정하는 maximal `root_joint`를 만든다. 이 joint는 articulation에 속하지 않으므로, `write_root_pose_to_sim`으로 손 기울기나 물체 위치를 바꿔도 반영되지 않는다.
   - 증상: 매 step reset이 일어났다. 손끝과 핸들 거리가 0.16m로 기준을 넘었다.
   - 해결: USD에서 `root_joint`를 비활성화하고 `fix_root_link=True`로 설정한다.
4. **nested rigid body라 contact sensor가 붙지 않았다.**
   - `activate_contact_sensors`는 최상위 body에만 적용된다.
   - 해결: `_setup_scene`에서 물체의 모든 link에 `PhysxContactReportAPI`를 직접 적용한다.
5. **contact offset 설정이 instanced prim에 적용되지 않았다.**
   - 해결: 변환 단계에서 asset 파일에 contact offset 0.002, rest offset 0.0을 직접 기록한다. 원본 sim 설정과 같은 값이다.
6. **USD 출력 폴더 이름이 충돌했다.**
   - `driver/0001_stripe`와 `boxnut/0001_stripe`가 같은 이름이다.
   - 해결: 출력을 `assets/usd/<subset>/`로 분리했다.
7. **학습 중에 viewer를 띄우면 GPU 메모리가 부족했다.**
   - 원인: 8192 env용 PhysX GPU 버퍼를 viewer도 그대로 잡는다.
   - 해결: env가 256개 이하면 버퍼를 1/16로 줄인다. 8192 env 학습 설정은 바꾸지 않았다.
8. **`SimulationApp.close()`가 종료 시 멈춘다.**
   - 해결: `train_lab.py`를 비롯한 스크립트가 `os._exit`로 종료한다.

## 4. 원본과의 차이 (의도적 / 불가피)

| 항목 | IsaacGym 원본 | Isaac Lab 포팅 |
|---|---|---|
| 물체 scale | bucket `list[i % S]` ± 0.025 연속 jitter | jitter 없이 bucket 값만 사용 (articulation은 런타임에 scale을 바꿀 수 없음. `MultiAssetSpawnerCfg(random_choice=False)`로 env에 배정) |
| 물체 종류 (screwdriver 0000/0001) | env마다 무작위 | env 번호로 결정 (균등 배분) |
| reset 시 screw/nut joint 각도 | reset하지 않아 episode를 넘어 누적 | 0으로 reset (628 rad limit에 닿는 문제 방지) |
| 손 effort limit | DOF `effort` 속성으로 clamp | Python에서 같은 값으로 명시적 clamp |
| nut joint limit | nut URDF에 `<limit>` 없음 | importer가 [0, 628.3] rad로 설정 |
| 물리 엔진 | PhysX 4 | PhysX 5 (solver 반복 횟수, offset, damping 값은 복사했지만 접촉 동역학은 동일하지 않음) |
| student env 수 | 48 | 1024 (사용자 결정. 48 env는 약 1.1k FPS로 너무 느림) |

위 항목 외에는 원본 로직을 1:1로 옮겼다. 대상은 다음과 같다.

- reward와 termination
- observation과 `priv_info` 레이아웃
- curriculum
- domain randomization 전체: PD gain, noise, random force, friction, mass, COM

내부적으로 IsaacGym 규약(xyzw 쿼터니언, env-local 좌표)을 유지하므로 `priv_info`의 의미도 원본과 같다.

검증한 내용:

- `priv_info_dim`이 원본 정의에서 계산한 값과 같다 (screwdriver 84, nut-bolt 127).
- obs 96, proprio_hist (30, 24).
- URDF forward kinematics로 계산한 손끝–물체 거리와 sim에서 잰 거리가 일치한다 (약 0.06m).

## 5. 학습 결과

학습은 모두 seed 42, RTX 5080 한 장에서 진행했다.

### 5.1 Screwdriver teacher (`outputs/XHandHoraScrewDriver_teacher/lab_run1`)

- 8192 envs, 약 42–48k FPS, GPU 메모리 약 11.4GB.
- **약 231M step에서 사용자 요청으로 중단했다.** 1B 예산이었다.

| step | 10M | 26M | 69M | 231M |
|---|---|---|---|---|
| episode reward | 25 | 332 | 300 | 284 |
| episode 길이 | 62 | 250 | 225 | 246 |
| 핸들 각속도 (rad/s) | 0.13 | 0.83 | 0.83 | 0.66 |

- 26M 이후 reward가 300~430 사이에서 정체했다.
- 체크포인트: `last.pth`(196M 시점), `best_reward_428.60.pth`(약 50M 시점).
- student는 아직 학습하지 않았다.

### 5.2 Nut-bolt teacher (`outputs/XHandHoraNutBolt_teacher/lab_run1`)

- 대상 물체 trinut, 8192 envs, 약 40–48k FPS. **1B step을 완료했다** (약 7시간).

학습 곡선 (탐색 noise 포함):

| step | 10M | 100M | 200M | 600M | 1000M |
|---|---|---|---|---|---|
| episode reward | 246 | 1563 | 1759 | 1851 | 2017 |
| episode 길이 (최대 800) | 161 | 469 | 498 | 515 | 515 |
| nut 접촉 비율 | 0.96 | 0.94 | 0.97 | 0.98 | 0.98 |

탐색 noise 없는 결정적 평가 (256 envs × 1600 steps, teacher 조건 `reset_dist_threshold=0.07`):

| 체크포인트 | reward | 길이 | 800 step 완주 | 회전량/episode | 각속도 | 정체로 종료 | 손가락 이탈 종료 |
|---|---|---|---|---|---|---|---|
| 98M | 1513 | 430 | 42% | 12.1 rad | 0.56 | 49% | 8.3% |
| 196M | 1789 | 491 | 50% | 14.6 rad | 0.59 | 45% | 5.1% |
| 491M | 1585 | 477 | 48% | 13.3 rad | 0.55 | 49% | 3.0% |
| 983M | 1999 | 495 | 50% | 15.8 rad | 0.63 | 47% | 2.9% |
| **best (2030.76)** | **2032** | **516** | **52%** | **16.2 rad** | **0.63** | 45% | 2.5% |

- nut을 놓치는 일은 거의 없다. 이탈은 2.5%, 접촉 상실은 0%다.
- 약 절반의 episode는 40초(800 step)를 끝까지 돈다.
- 나머지 약 45%는 nut이 3.5초(70 step) 동안 거의 움직이지 않아 정체 조건으로 종료된다.
- 200M 이후 개선 폭은 작다.

### 5.3 Nut-bolt student (`outputs/XHandHoraNutBolt_student_padapt/lab_run1`)

- teacher `best_reward_2030.76.pth`에서 distillation했다.
- 1024 envs, 약 15k FPS. **약 68M step에서 수렴해 중단했다** (약 1.2시간).

| step | 4M | 21M | 41M | 68M |
|---|---|---|---|---|
| bc_loss | 0.90 | 0.68 | 0.62 | 0.59 |
| latent_loss | 0.147 | 0.132 | 0.130 | 0.129 |
| episode reward | 979 | 1422 | 1569 | 1613 |
| episode 길이 | 316 | 410 | 440 | 449 |

teacher와 student 비교 (결정적 평가, 256 envs × 1600 steps):

| 정책 | 평가 조건 | reward | 길이 | 완주 | 회전량/episode | 각속도 | 정체 | 이탈 |
|---|---|---|---|---|---|---|---|---|
| teacher | student 조건 (dist 0.12, obs noise ↑) | 2318 | 528 | 54% | 16.4 rad | 0.62 | 46% | 0% |
| student | student 조건 | 1506 | 417 | 38% | 10.6 rad | 0.51 | 62% | 0% |
| teacher | teacher 조건 (dist 0.07) | 1907 | 494 | 49% | 15.2 rad | 0.60 | 50% | 1.4% |
| student | teacher 조건 | 1361 | 420 | 39% | 10.9 rad | 0.52 | 55% | 6.1% |

- student는 privileged 정보 없이 관절 기록만 쓴다. 성능은 teacher의 약 65%다 (reward와 회전량 모두).
- 격차는 대부분 정체 종료가 늘어난 데서 온다 (46% → 62%).
- 체크포인트: `stage2_nn/model_best.ckpt`(약 68M), `model_last.ckpt`(약 60M, 위 평가에 사용).

### 5.4 해석 시 주의

- 원본 논문과 레포에 시뮬레이션 학습 곡선이나 성능 수치가 없다. 그래서 위 수치가 원본 IsaacGym 재현 수준인지는 판단할 수 없다.
- 4절의 차이 중 특히 물리 엔진 버전(PhysX 4 → 5)과 scale jitter 제거가 성능에 영향을 줬을 수 있다.
- 학습 로그의 각속도(`screw/angular_velocity`)와 평가 스크립트의 각속도는 계산 방식이 다르다. 학습 로그는 탐색 noise가 섞인 순간 joint 속도이고, 평가는 control step 단위 평균이다.

## 6. 재현 명령

```bash
cd ~/IsaacLab/projects/dexscrew
PY=~/env_isaaclab6/bin/python

# 0) asset 변환 (1회)
$PY scripts/convert_urdf_to_usd.py

# 1) teacher
scripts/lab_nutbolt_teacher.sh 0 42 lab_run1 train.ppo.max_agent_steps=1000000000
scripts/lab_screwdriver_teacher.sh 0 42 lab_run1 train.ppo.max_agent_steps=1000000000

# 2) student (수렴하면 직접 중단)
scripts/lab_nutbolt_student_padapt.sh 0 42 lab_run1 task.env.numEnvs=1024 \
    checkpoint=outputs/XHandHoraNutBolt_teacher/lab_run1/stage1_nn/best_reward_2030.76.pth

# 평가
$PY scripts/eval_policy.py --task XHandHoraNutBolt --num_envs 256 --steps 1600 \
    task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.07 \
    --ckpts outputs/XHandHoraNutBolt_teacher/lab_run1/stage1_nn/best_reward_2030.76.pth
$PY scripts/eval_policy.py --task XHandHoraNutBolt --num_envs 256 --steps 1600 \
    task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.12 \
    task.env.randomization.obs_noise_t_scale=0.01 task.env.randomization.obs_noise_e_scale=0.02 \
    --student --ckpts outputs/XHandHoraNutBolt_student_padapt/lab_run1/stage2_nn/model_last.ckpt

# GUI viewer (학습 중에도 사용 가능, 최신 체크포인트를 다시 로드)
$PY scripts/watch_policy.py --task XHandHoraNutBolt --run outputs/XHandHoraNutBolt_teacher/lab_run1 \
    task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.07

# 학습 곡선
~/env_isaaclab6/bin/tensorboard --logdir outputs --port 6006
```

`outputs/`와 `assets/usd/`는 git에서 제외되어 있다. `assets/usd/`는 변환 스크립트로 다시 만들 수 있다. 체크포인트는 로컬 `outputs/`에만 있다.

## 7. 남은 작업

1. **screwdriver teacher 재개 또는 재학습, 이어서 student 학습.**
   - `checkpoint=...last.pth`로 재개하면 weight와 normalizer만 로드된다.
   - step 카운터, optimizer, 각속도 penalty curriculum(30–60M)은 처음부터 다시 시작한다.
   - 완전히 이어서 학습하려면 이 부분을 손봐야 한다.
2. **nut teacher의 정체 종료(약 45%) 원인 분석.** 정책 한계인지, 포팅 물리 차이(마찰, 접촉) 때문인지 viewer로 확인한다.
3. **boxnut / crossnut 학습.** `task.env.object.type=screw_boxnut` 등으로 지정한다.
4. **실기 배포 단계 포팅.** `scripts/convert_student_jit.sh`와 `xhand-deploy`가 Isaac Lab 체크포인트와 호환되는지 확인한다.
