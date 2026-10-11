# 넛 크기 일반화 학습과 DexGen 데이터 수집 (2026-10-10 ~ 10-11)

[isaaclab_reproduction_report.md](isaaclab_reproduction_report.md)의 재현 결과에 이어지는 작업이다. 두 정책 원천을 넛 크기에 일반화하도록 다시 학습했고, 그 정책들로 DexGen 학습 데이터 1e8 transition을 모았다.

- **dexscrew nut-bolt:** 이 레포
- **dexgen 자유 너트 goal task:** `../dexgen`

| 문서 | 내용 |
|---|---|
| `../dexgen/docs/training_log_2026-10-10.md` | 30분 간격 점검 기록, 모든 판단과 수치 |
| `../dexdata/docs/collection_guide.md` | 수집에 쓴 정책, 필수 설정, 명령, 수집량, 시간 |
| `../dexgen/docs/plan_review_and_revision.md` | DexGen 계획 검토와 결정 사항 D1–D6 |

## 1. 범위 (사용자 결정)

- **dexgen:** 넛 크기 0.7–1.3. 모양 랜덤화는 하지 않는다. 질량 랜덤화는 기존(D2) 그대로 둔다.
- **dexscrew:** 크기 0.7–1.3을 목표로 했지만, 1.3은 물리가 폭주해서 **0.7–1.25**로 학습했다 (3.1절). 모양은 trinut 하나만 쓴다.
- **고정한 것:** D2 DR 범위, task 정의, PPO 설정(dexgen: std 0.3 고정, entropy 0, LR 2e-4 고정, 1024 env)

## 2. dexgen 자유 너트

### 2.1 겪은 문제와 해결

| 단계 | 문제 | 원인 | 조치 |
|---|---|---|---|
| 크기 랜덤화 | 이전 run은 사실상 고정 크기로 학습됨 | `XformCommonAPI.SetScale`이 기존 scale op와 맞지 않음 | `xformOp:scale`에 직접 기록 |
| 처음부터 크기 혼합 학습 (v1/v2/v3/v5 grasp 표, A/B run) | hold-only: 목표 도달 0.03–0.08 (기준 run은 iter 200에 10) | 아래 두 행 참고 | fine-tune으로 전환 |
| robust grasp 표 | settle 후 tilt 22–38° | 이 손으로 0-action에서 똑바로 버티는 grasp가 사실상 없음. 기존 1.0 정책도 0-action에서는 42°까지 기운다 | grasp 표를 쓰지 않음 |
| 첫 목표 위치 | grasp가 자리 잡는 동안 넛이 9–14 mm 이동하는데 위치 허용 오차는 12 mm | 목표 위치를 settle 전에 지정 | `goal_settle_steps` 옵션 추가 (최종 정책에서는 쓰지 않음) |
| grasp 탐색 v3 | 떨어지는 grasp를 1등으로 선택 | 떨어진 env는 직립으로 리셋돼 tilt가 0°로 읽힘 | 버틴 반복에서만 tilt를 재도록 수정 |

**최종 방법:** 고정 크기 정책에서 fine-tune했다. 설정은 `--checkpoint` + `--cfg "scale_range=(0.7,1.3)"`이고, 초기 grasp는 INIT_Q, DR은 full로 이어 학습했다.

### 2.2 최종 정책과 크기별 성능

| 정책 | checkpoint (`../dexgen/outputs/xhand_nut_goal/`) |
|---|---|
| 일반 | `2026-10-10_23-26-04_goal_scale0713_ft_initq/model_14250.pt` (`goal_contact2/model_9999`에서 4250 iter) |
| slow (fingertip speed 15) | `2026-10-11_03-53-58_goal_scale0713_ft_slowstyle15/model_13998.pt` (`goal_contact2_slowstyle15/model_9999`에서 4000 iter) |

timeout까지 버틴 비율. 명목 물리, 64 env, 800 step으로 쟀다.

| 크기 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 |
|---|---|---|---|---|---|---|---|
| 일반 | 98% | 91% | 95% | 98% | 63% | **12%** | 63% |
| slow | 85% | 86% | 90% | 95% | 68% | **11%** | 63% |
| 이전 1.0 고정 정책 | 1% | 25% | 87% | 99% | 64% | 11% | 45% |

- **남은 한계:** 1.15–1.25 구간이 약하다.
  - 두 정책 모두 1.2에서만 무너진다. 그래서 정책보다 이 크기에서의 INIT_Q grasp가 원인으로 보인다.
  - v2 grasp 표로 보완하려던 fine-tune은 실패했다. tilt가 큰 grasp에서 시작하면 정책이 넛을 돌리지 못했다.

### 2.3 코드 변경 (`../dexgen`, git 관리 밖)

| 파일 | 변경 |
|---|---|
| `dexgen/nut_goal_env.py` | 크기 적용 버그 수정 |
| | `scale_grasp_table`, `scale_jitter` 옵션 (크기 버킷별 초기 grasp) |
| | grasp 표의 `settled` 상태로 리셋 |
| | `goal_settle_steps` (모두 cfg 옵션이고 기본값은 기존 동작과 같다) |
| `train.py` | `--learning_rate`, `--lr_schedule`, `--cfg KEY=VALUE` |
| `scripts/find_scale_grasps.py` | 크기별 grasp 탐색 (2단계 robust, tilt 제한, settled 상태 저장) |
| `scripts/measure_scale_feasibility.py` | 크기별 평가 (`--grasp_table`, `--goal_settle_steps`) |
| `scripts/watch_policy.py` | `--scale_range MIN MAX` |
| `scripts/train_status.py` | TensorBoard 요약 |

## 3. dexscrew nut-bolt (이 레포)

### 3.1 env 변경

- **`task.env.keepNutHeight` (기본 False)**
  - 넛볼트 asset은 base를 기준으로 통째로 scale된다. 그래서 크기가 바뀌면 넛 높이도 바뀌어 손과 맞지 않게 된다.
  - 이 옵션을 켜면 넛 joint 높이(`nutJointHeight`, 기본 0.1 m)가 모든 크기에서 같도록 asset을 위아래로 옮긴다.
  - 새 config 키라서 hydra에서는 `+task.env.keepNutHeight=True`로 준다.
- **크기 1.3을 뺀 이유**
  - 1.3에서는 리셋 직후 1–2 step 만에 넛 joint가 1000–3700 rad/s로 튀고, joint 범위 밖(음수)까지 간다.
  - 0-action에서는 안정적이다. 손가락이 넛을 움직이는 순간 접촉 solver가 무너지는 것으로 보인다.
  - 넛 윗면 기준으로 높이를 맞춰 봤고, joint 속도 제한(20 rad/s)도 걸어 봤지만 둘 다 해결하지 못했다. 1.25까지는 폭주가 없다.
- **`scripts/watch_policy.py --student`:** ProprioAdapt student checkpoint(`stage2_nn/*.ckpt`)를 GUI로 본다.

### 3.2 teacher

- **run:** `outputs/XHandHoraNutBolt_teacher/lab_scale07125`
  - `lab_run1/best_reward_2030.76.pth`에서 이어 학습했다.
  - 크기 `[0.7,0.8,0.9,1.0,1.1,1.2,1.25]`, `keepNutHeight`, **`train.ppo.learning_rate=1.3e-4`**
- **학습률을 1.3e-4로 맞춘 이유:**
  - 기본값 5e-3으로 이어 학습하면 optimizer가 새로 시작해서, 첫 몇 번의 업데이트에서 정책이 망가진다. 이때 far 종료가 66%였다.
  - 1.3e-4는 lab_run1이 마지막에 쓰던 학습률이다.
- **411M에서 중단했다.** 300M 이후 entropy가 계속 줄면서 reward가 2회 연속 떨어졌다 (1257 → 1053).

rotation/episode (rad). 256 env, 1000 step.

| 크기 | 0.7 | 0.85 | 1.0 | 1.15 | 1.2 | 1.25 |
|---|---|---|---|---|---|---|
| lab_run1 teacher (0.95–1.05) | 6.8 | 15.6 | 14.5 | 12.9 | 0.9* | 0.1* |
| `best_reward_1413.08` | 6.8 | 12.4 | 15.1 | 16.7 | 14.5 | 1.7 |
| `ep_2500` (245M) | 11.1 | 13.0 | 12.5 | 14.7 | 11.8 | 1.7 |

(* 64 env)

### 3.3 student (ProprioAdapt, 1024 env)

| student | teacher | 0.7 | 0.85 | 1.0 | 1.15 | 1.2 | 1.25 | 평균 |
|---|---|---|---|---|---|---|---|---|
| `lab_scale07125` | `ep_2500` | 4.3 | 6.7 | 5.1 | 7.7 | 7.0 | 1.1 | 5.3 |
| **`lab_scale07125_t1413`** (사용) | `best_reward_1413.08` | 4.1 | 7.2 | 8.7 | 10.8 | 10.8 | 1.3 | **7.2** |
| lab_run1 (0.95–1.05) | lab_run1 best | 4.4 | 7.2 | 12.2 | 7.4 | 8.1 | 1.2 | 6.8 |

- 크기가 섞이면 student가 teacher를 따라 하는 비율이 낮아진다. 1.0 기준으로 lab_run1은 84%, t1413은 58%다.
- proprio 이력만으로 크기까지 추정해야 하기 때문으로 보인다.
- 1.0 근처 회전 품질은 lab_run1 student가 더 좋다.

실행 예:

```bash
# teacher 이어 학습
scripts/lab_nutbolt_teacher.sh 0 42 lab_scale07125 \
  checkpoint=outputs/XHandHoraNutBolt_teacher/lab_run1/stage1_nn/best_reward_2030.76.pth \
  "task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]" +task.env.keepNutHeight=True \
  train.ppo.max_agent_steps=600000000 train.ppo.learning_rate=1.3e-4
# student
scripts/lab_nutbolt_student_padapt.sh 0 42 lab_scale07125_t1413 task.env.numEnvs=1024 \
  checkpoint=outputs/XHandHoraNutBolt_teacher/lab_scale07125/stage1_nn/best_reward_1413.08.pth \
  "task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]" +task.env.keepNutHeight=True
# 크기별 평가
~/env_isaaclab6/bin/python scripts/eval_policy.py --student --num_envs 256 --steps 1000 \
  task.env.object.type=screw_trinut task.env.reset_dist_threshold=0.07 \
  "task.env.randomization.randomizeScaleList=[1.0]" +task.env.keepNutHeight=True \
  --ckpts outputs/XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt
```

## 4. DexGen 데이터 수집 (`../dexdata/data/v1`, 2026-10-11)

명령은 `../dexdata/logs/collect_v1.sh`, 방법은 `../dexdata/docs/collection_guide.md`에 있다.

| run | `policy_name` | transition | episode | 정제 후 남은 비율 | 시간 |
|---|---|---|---|---|---|
| `nut_goal_normal_s0` | `nut_goal_normal` | 40.0M | 169,736 | 87% | 45분 |
| `nut_goal_slow_s0` | `nut_goal_slow15` | 20.0M | 96,158 | 84% | 24분 |
| `nutbolt_student_s0` | `nutbolt_student_scale` | 40.0M | 143,115 | 99% | 41분 |
| 합계 | | **100.0M** (91 GB) | 409,009 | | 1시간 50분 |

- **검증:** `validate_dataset.py` 모든 검사를 통과했다.
  - FK keypoint 위치 최대 0.001 mm, 자세 최대 0.1°
  - 시간 정렬은 정확히 일치한다.
- **dexgen 수집에서는 `--cfg dr_ramp_steps=0`이 필수다.**
  - 이게 없으면 env가 DR을 0부터 ramp해서, 첫 48000 env step 동안 질량, 마찰, 손목 ±30°가 모두 명목값이다.
  - 이번 수집 데이터의 질량은 0.03–0.25로, full DR이 적용된 것을 확인했다.
- **크기 분포:** 정제 후에도 균등 분포에 가깝다.
  - dexgen: 0.7–0.85 25%, 0.85–1.05 33%, 1.05–1.15 16%, 1.15–1.22 11%, 1.22–1.3 14%
  - dexscrew: 7개 버킷이 고르게 들어갔다.

## 5. 교훈

1. 처음부터 크기를 섞어 학습하면 hold-only local optimum에 빠진다. 고정 크기 정책에서 fine-tune해야 한다.
2. "robust하게 버티는" grasp가 upright goal에는 오히려 해가 될 수 있다. grasp 품질은 tilt까지 함께 봐야 한다.
3. PPO를 이어 학습할 때는 학습률을 이전 run의 마지막 값으로 맞춘다.
4. DR curriculum이 있는 env로 데이터를 수집할 때는 ramp를 끈다.
5. Isaac Sim: `SimulationApp.close()`가 멈출 수 있으므로, 학습 process를 끝낼 때는 kill -9로 확인한다.
6. shell: `pkill -f` 패턴이 자기 자신의 명령줄과 겹치지 않게 한다.
