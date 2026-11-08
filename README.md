# ROVER: Reward-free Pretraining via Occupancy Coverage Maximization

Official implementation of **Reward-free Pretraining for Reinforcement Learning via Occupancy
Coverage Maximization** (Pratticò, Novelli, Pontil, Ciliberto; NeurIPS 2026).

ROVER pretrains a *transferable exploration policy* without any reward. It minimises the squared
RKHS norm of the discounted state occupancy, `J(pi) = ||mu_{d^pi}||^2_H` (equivalently, it maximises the
metric Renyi-2 entropy of the occupancy and, in tabular problems, matches the uniform measure in MMD),
using a learned **resolvent world model**: a kernel-ridge-regression estimate of the transition operator
in a learned feature space, a closed-form gradient of `J` through the resolvent, policy mirror descent
(PMD) in closed form, and a virtual **sink state** that makes unsupported regions costly and prevents
expansion-collapse cycles.

## Algorithm in one page

Alternate data collection (epsilon-greedy over the current policy) with

1. **Representation learning** -- L1-normalised features `phi(x)` trained with InfoNCE on `(x, a, x')` (Eq. 9);
2. **World model** -- sink-biased kernel ridge regression `T = Phi^T G Psi`, `G = (K + lambda I)^-1` (Eq. 17);
3. **Policy mirror descent** -- `pi_{t+1}(.|x) = softmax(-eta <psi(x, .), Psi^T sum_s c_s>)` with the dual gradient
   coefficients `c` of Thm. 2 (line-searched step size, best-iterate restoration).

Two variants are implemented:

| Variant | Used for | Code | Kernel(s) |
| --- | --- | --- | --- |
| **Standard ROVER** | gridworlds (Middle Room, Multi-Rooms, Maze, Two Rooms; one-hot and pixel observations) | `agent/rover.py`, `agent/rover_math.py` | inner product on learned features, all `n` transitions |
| **Subspace ROVER** | PointMaze (App. E, G.8, Table 1) | `agent/rover_subspace.py`, `agent/nystrom.py`, `agent/landmarks.py` | Gaussian *dynamics* kernel on learned full-state features; Gaussian *coverage* kernel on the projection `g(s) = (x, y)`; Nystrom landmarks chosen by GPU pivoted Cholesky |

Subspace ROVER measures coverage of the pushforward `g_# d^pi` while the world model still sees the full state
(e.g. velocity). With `g = id` and the same kernel it reduces to standard ROVER (this is unit-tested).
Details and verification: [`docs/implementation_summary.md`](docs/implementation_summary.md).

## Installation

```bash
conda env create -f environment.yml     # Python 3.11, PyTorch (CUDA), Gymnasium(-Robotics), MuJoCo, MiniGrid, Hydra
conda activate rover
export MUJOCO_GL=egl                    # headless rendering for PointMaze pixel frames
python -m pytest tests -q               # optional: unit tests (GPU tests are skipped without CUDA)
```

Runs use one GPU (`device=cuda`); override `device=cpu` for smoke tests.

## Repository structure

```
agent/rover.py              standard ROVER agent (encoder InfoNCE + dense PMD actor)
agent/rover_math.py         occupancy objective, sink Gram and Thm. 2 gradient (factorisations cached)
agent/rover_subspace.py     subspace ROVER agent (state observations)
agent/nystrom.py            block-wise Nystrom occupancy model + reduced-rank gradient
agent/landmarks.py          GPU pivoted-Cholesky landmark selection
agent/kernels.py            Gaussian / inner-product kernels, median bandwidth
agent/rover_networks.py     encoders and transition projector
agent/{rnd,smm,icm_apt,maxent,cic,...}  reward-free baselines;  agent/{ddpg,sac,cql}*  downstream learners
env/                        Multi-room / Maze / Middle-Room gridworlds, PointMaze wrapper (goal-hidden observations)
configs/agent/              agent configs (rover_paper = standard ROVER of the paper)
configs/scripts/            one YAML per paper experiment (see matrix below)
pretrain.py, train.py       gridworld pretraining; downstream finetuning / offline training
pretrain_pointmaze.py       subspace-ROVER PointMaze pretraining;  evaluate_pointmaze.py  coverage evaluation
scripts/                    run wrappers and replay-buffer coverage analysis
tests/                      gradient, Cholesky, Nystrom, goal-leakage, coverage and config tests
docs/                       implementation summary, reproduction report, handoff notes
```

## Training and evaluation commands

### Gridworlds (standard ROVER)

```bash
python pretrain.py --config-name scripts/main_text/middle_room_behavior         # Fig. 2 / 7 (reward-free behaviour)
python pretrain.py --config-name scripts/main_text/multirooms_coverage_discrete seed=1   # Fig. 3, one-hot
python pretrain.py --config-name scripts/main_text/maze_coverage_pixels seed=1           # Fig. 3, 84x84 RGB
scripts/run_gridworld.sh main_text/multirooms_coverage_discrete 0 "1 2 3 4 5"            # five seeds on GPU 0

# Samples needed to reach 25/50/75/90/95/99/100 % state coverage, from the saved replay buffer:
python scripts/coverage_speed.py exp_local/<date>/<run_dir>
```

Baselines use the same configs: append `agent=rnd_discrete | smm_discrete | icm_apt_discrete | maxent_discrete | cic_discrete`.
Downstream transfer (Fig. 8-10) and offline replay transfer (Table 3):

```bash
python train.py --config-name scripts/appendix/two_rooms_finetune_ddpg p_path=/path/to/snapshot.pt
python train_offline.py --config-name scripts/appendix/offline_two_rooms_cql replay_buffer_dir=/path/to/pretraining/buffer
```

### PointMaze (subspace ROVER)

```bash
python pretrain_pointmaze.py --config-name scripts/pointmaze/umaze_states seed=1 'env.pointmaze.start_position=[-1.0,1.0]'
python evaluate_pointmaze.py results/pointmaze/umaze_states/start_-1.0_1.0/seed1      # 3 evaluation seeds x 50 trajectories
scripts/run_pointmaze_umaze.sh 0 "1 2 3"          # the full Table-1 protocol: 3 starts x 3 seeds, GPU 0
```

Coverage = percentage of free cells of a 90x90 maze grid (radius 0.08) visited by 50 stochastic-policy
trajectories of 300 steps (the policy is evaluated epsilon-greedy, exactly as it explores).
The three start positions are `[-1,1]`, `[1,1]`, `[1,0]`.

### Reproducibility Matrix

| Paper result | Experiment | Environment | Observation | Algorithms | Config YAML | Entry command |
| --- | --- | --- | --- | --- | --- | --- |
| Sec. 4.1, Fig. 2 | Reward-free behavior snapshots | Middle Room | One-hot states | ROVER, RND, SMM, ICM-APT, MAXENT, CIC | `configs/scripts/main_text/middle_room_behavior.yaml` | `python pretrain.py --config-name scripts/main_text/middle_room_behavior` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Multi-Rooms | One-hot states | ROVER and baselines | `configs/scripts/main_text/multirooms_coverage_discrete.yaml` | `python pretrain.py --config-name scripts/main_text/multirooms_coverage_discrete` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Multi-Rooms | 84x84 RGB | ROVER and pixel-capable baselines | `configs/scripts/main_text/multirooms_coverage_pixels.yaml` | `python pretrain.py --config-name scripts/main_text/multirooms_coverage_pixels` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Maze | One-hot states | ROVER and baselines | `configs/scripts/main_text/maze_coverage_discrete.yaml` | `python pretrain.py --config-name scripts/main_text/maze_coverage_discrete` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Maze | 84x84 RGB | ROVER and pixel-capable baselines | `configs/scripts/main_text/maze_coverage_pixels.yaml` | `python pretrain.py --config-name scripts/main_text/maze_coverage_pixels` |
| App. G.3, Fig. 8 | Reward-free pretraining for transfer | Two Rooms | One-hot states | ROVER and baselines | `configs/scripts/appendix/two_rooms_pretrain.yaml` | `python pretrain.py --config-name scripts/appendix/two_rooms_pretrain` |
| App. G.3, Fig. 8 | Online policy transfer | Two Rooms | One-hot states | DDPG, DDPG + pretrained policy | `configs/scripts/appendix/two_rooms_finetune_ddpg.yaml` | `python train.py --config-name scripts/appendix/two_rooms_finetune_ddpg p_path=/path/to/snapshot.pt` |
| App. G.4-G.6, Fig. 9-10, Table 2 | Reward-free pretraining for transfer/scaling | Multi-Rooms | One-hot states | ROVER and baselines | `configs/scripts/appendix/multirooms_pretrain.yaml` | `python pretrain.py --config-name scripts/appendix/multirooms_pretrain` |
| App. G.4-G.5, Fig. 9-10 | Online DDPG transfer | Multi-Rooms | One-hot states | DDPG, DDPG + pretrained policy | `configs/scripts/appendix/multirooms_finetune_ddpg.yaml` | `python train.py --config-name scripts/appendix/multirooms_finetune_ddpg p_path=/path/to/snapshot.pt` |
| App. G.4, Fig. 9 | Online SAC transfer | Multi-Rooms | 84x84 RGB | SAC, SAC + pretrained policy | `configs/scripts/appendix/multirooms_finetune_sac.yaml` | `python train.py --config-name scripts/appendix/multirooms_finetune_sac p_path=/path/to/snapshot.pt` |
| App. G.7, Table 3 | Offline replay-buffer transfer | Two Rooms | One-hot states | CQL | `configs/scripts/appendix/offline_two_rooms_cql.yaml` | `python train_offline.py --config-name scripts/appendix/offline_two_rooms_cql replay_buffer_dir=/path/to/pretraining/buffer` |
| App. G.7, Table 3 | Offline replay-buffer transfer | Multi-Rooms | One-hot states | CQL | `configs/scripts/appendix/offline_multirooms_cql.yaml` | `python train_offline.py --config-name scripts/appendix/offline_multirooms_cql replay_buffer_dir=/path/to/pretraining/buffer` |
| App. G.7, Table 3 | Offline replay-buffer transfer | Two Rooms | One-hot states | Offline DDPG reference | `configs/scripts/appendix/offline_two_rooms_ddpg.yaml` | `python train_offline.py --config-name scripts/appendix/offline_two_rooms_ddpg replay_buffer_dir=/path/to/pretraining/buffer` |
| App. G.7, Table 3 | Offline replay-buffer transfer | Multi-Rooms | One-hot states | Offline DDPG reference | `configs/scripts/appendix/offline_multirooms_ddpg.yaml` | `python train_offline.py --config-name scripts/appendix/offline_multirooms_ddpg replay_buffer_dir=/path/to/pretraining/buffer` |
| Sec. 4.3, Table 1 | Position coverage (subspace ROVER) | PointMaze U-Maze | (x, y, vx, vy) states | ROVER, RND, SMM, ICM-APT, MAXENT, CIC | `configs/scripts/pointmaze/umaze_states.yaml` | `python pretrain_pointmaze.py --config-name scripts/pointmaze/umaze_states seed=1` |


The default pretraining configs use `agent=rover_paper`. Goal locations are changed by selecting another
environment config, e.g. `env=gridworld/multiplerooms5_4x4_1`; older appendix variants are in `configs/env/gridworld/appendix/`.

## Hyperparameters

Defaults live in `configs/agent/rover_paper.yaml` (standard ROVER, each parameter is commented with its paper symbol)
and in `configs/scripts/pointmaze/umaze_states.yaml` (subspace ROVER). Main values:

| Parameter | Standard ROVER | Subspace ROVER (PointMaze) | Meaning |
| --- | --- | --- | --- |
| `discount` (gamma) | 0.9 (Middle Room), 0.99 (others) | 0.99 | occupancy discount |
| `lambda_reg` (lambda) | 1e-3 | 1e-3 | KRR ridge |
| `lr_actor` (eta) | 10 | 25 | PMD step size (initial value of the line search) |
| `pmd_steps` | 250 | 5 | PMD iterations per actor update |
| `sink_schedule` (eps) | `linear(0, 0.05, 50k)` | `linear(0, 5, 50k, 250k)` | sink scale |
| `batch_size_actor` | 5000 (-> 4096 transitions) | 100 000 latest transitions | support of the world model |
| `subsamples` / `pca_truncation` | - | 10 000 / 1000 | Nystrom landmarks / rank of the adjoint solve |
| `feature_dim`, `hidden_dim` | 200, 1024 | 256, 1024 | encoder |
| `epsilon_schedule` | 0.15 | 0.3 | epsilon-greedy |
| `update_actor_every_steps` | 1500 | 20 000 | actor update period (frames) |

| Algorithm | Role in paper experiments | Main hyperparameters |
| --- | --- | --- |
| ROVER | Reward-free occupancy-coverage pretraining | `discount=0.99`, `lr_actor=10`, `pmd_steps=250`, `pmd_eta_mode=backtracking`, `sink_schedule=linear(0.0, 0.05, 50_000)`, `batch_size_actor=5_000`, `T_init_steps=300`, `lr_T=1e-3`, `lr_encoder=1e-3`, `feature_dim=200`, `batch_size=1024`, `nstep=1` |
| RND | Reward-free exploration baseline | `rnd_lr=1e-4`, `actor_lr=1e-4`, `critic_lr=1e-4`, `critic_target_tau=0.01`, `rnd_rep_dim=512`, `rnd_scale=1.0`, `eps_schedule=0.25`, `batch_size=1024`, `nstep=1` |
| SMM | State-marginal matching baseline | `z_dim=4`, `sp_lr=1e-3`, `vae_lr=1e-2`, `vae_beta=0.5`, `state_ent_coef=1.0`, `latent_ent_coef=1.0`, `latent_cond_ent_coef=1.0`, `actor_lr=1e-4`, `critic_lr=1e-4`, `batch_size=1024`, `nstep=3` |
| ICM-APT / APT | Particle-entropy and curiosity baseline | `icm_lr=1e-4`, `actor_lr=1e-4`, `critic_lr=1e-4`, `icm_rep_dim=512`, `icm_scale=1.0`, `knn_k=12`, `knn_avg=true`, `eps_schedule=0.25`, `batch_size=1024`, `nstep=3` |
| MAXENT | Entropy/KDE exploration baseline | `actor_lr=1e-4`, `critic_lr=1e-4`, `maxent_scale=1.0`, `maxent_eps=1e-3`, `maxent_bandwidth=0.1`, `maxent_kernel=epanechnikov`, `maxent_pca_components=32`, `maxent_rollout_every_steps=5000`, `batch_size=1024`, `nstep=1` |
| CIC | Skill-diversity exploration baseline | `lr=1e-4`, `actor_lr=1e-4`, `critic_lr=1e-4`, `feature_dim=1024`, `skill_dim=64`, `scale=1.0`, `update_skill_every_step=50`, `temp=0.5`, `batch_size=1024`, `nstep=3` |
| DDPG | Online finetuning and offline reference | `actor_lr=1e-7` for ROVER-initialized finetuning configs, `critic_lr=1e-4`, `critic_target_tau=0.01`, `update_every_steps=2`, `update_actor_after_critic_steps=7_999`, `feature_dim=200`, `batch_size=1024`, `nstep=1` |
| SAC | Pixel-based online finetuning | `actor_lr=1e-5`, `critic_lr=1e-5`, `alpha_lr=1e-5`, `init_temperature=0.1`, `critic_target_tau=0.01`, `actor_update_frequency=1`, `critic_target_update_frequency=1`, `feature_dim=200`, `batch_size=256`, `nstep=1` |
| CQL | Offline replay-buffer transfer | `lr=1e-4`, `critic_target_tau=0.01`, `n_samples=3`, `alpha=0.01`, `target_cql_penalty=5.0`, `epsilon_schedule=linear(1.0,0.1,100000)`, `batch_size=1024`, `nstep=1` |


## Outputs

Hydra writes single runs under `exp_local/<date>/...` (override with `hydra.run.dir=...`). Pretraining snapshots are
written under `models/${obs_type}/${domain}/${agent.name}/${seed}`, replay buffers (`save_buffer=true`) to `buffer/`.
PointMaze runs write `metrics.csv`, `final_snapshot.pt`, `results.json` (and `eval.json` after evaluation) to `output_dir`.
All of these are git-ignored. W&B logging is off by default (`use_wandb=true` to enable).

Check that a config composes without launching a run:

```bash
python pretrain.py --config-name scripts/main_text/middle_room_behavior --cfg job
```

## Reproducing the paper

`docs/reproduction_report.md` lists, for a representative subset of the experiments, the exact commands, the
reference numbers (marked as extracted from the paper or from the research logs) and the numbers measured with this
repository. Full-scale reproduction (all seeds, pixel PointMaze, all baselines) is not included.

## Known limitations

* Pixel PointMaze (Table 1, right column) is not part of this release.
* The goal-leakage fix changes observations relative to early research runs; results generated before the fix
  (research commit `eb935af`) are not directly comparable (see `docs/implementation_summary.md`).
* Subspace ROVER runs a single environment process; the reference runs used four parallel environments.
* The Nystrom normal equations use fixed 1e-6 ridge jitters; very smooth kernels amplify them.
* Only discrete action spaces are supported.

## Citation

```bibtex
@inproceedings{prattico2026rover,
  title     = {Reward-free Pretraining for Reinforcement Learning via Occupancy Coverage Maximization},
  author    = {Pratticò, Marco and Novelli, Pietro and Pontil, Massimiliano and Ciliberto, Carlo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
