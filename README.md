# ROVER

This repository contains code and Hydra configurations for reproducing the
experiments in **Reward-free Pretraining for Reinforcement Learning via
Occupancy Coverage Maximization**.

ROVER learns a transferable reward-free exploration policy by optimizing
occupancy coverage. The release is organized around the paper experiments:
main-text coverage diagnostics and appendix policy-transfer/offline-transfer
experiments.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate rover
```

Experiments were developed for Python 3.10 with CUDA-enabled GPUs. To run on
CPU, override `device=cpu`.

## Reproducing Paper Experiments

All paper-run YAMLs live under `configs/scripts/`. They are launched directly
through the existing Hydra entrypoints:

```bash
python pretrain.py --config-name scripts/main_text/middle_room_behavior
python pretrain.py --config-name scripts/main_text/multirooms_coverage_pixels seed=1
python train.py --config-name scripts/appendix/two_rooms_finetune_ddpg p_path=/path/to/snapshot.pt
```

To run multiple seeds, repeat the same command with `seed=1`, `seed=2`, etc.
The paper reports five seeds for online policy-transfer curves and seven seeds
for the offline transfer table.

### Reproducibility Matrix

| Paper result | Experiment | Environment | Observation | Algorithms | Config YAML | Entry command |
| --- | --- | --- | --- | --- | --- | --- |
| Sec. 4.1, Fig. 2 | Reward-free behavior snapshots | Middle Room | One-hot states | ROVER, RND, SMM, ICM-APT, MAXENT, CIC | `configs/scripts/main_text/middle_room_behavior.yaml` | `python pretrain.py --config-name scripts/main_text/middle_room_behavior` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Multi-Rooms | One-hot states | ROVER and baselines | `configs/scripts/main_text/multirooms_coverage_discrete.yaml` | `python pretrain.py --config-name scripts/main_text/multirooms_coverage_discrete` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Multi-Rooms | 84x84 RGB | ROVER and pixel-capable baselines | `configs/scripts/main_text/multirooms_coverage_pixels.yaml` | `python pretrain.py --config-name scripts/main_text/multirooms_coverage_pixels` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Maze | One-hot states | ROVER and baselines | `configs/scripts/main_text/maze_coverage_discrete.yaml` | `python pretrain.py --config-name scripts/main_text/maze_coverage_discrete` |
| Sec. 4.2, Fig. 3 | Full-coverage sample efficiency | Maze | 84x84 RGB | ROVER and pixel-capable baselines | `configs/scripts/main_text/maze_coverage_pixels.yaml` | `python pretrain.py --config-name scripts/main_text/maze_coverage_pixels` |
| App. F.3, Fig. 8 | Reward-free pretraining for transfer | Two Rooms | One-hot states | ROVER and baselines | `configs/scripts/appendix/two_rooms_pretrain.yaml` | `python pretrain.py --config-name scripts/appendix/two_rooms_pretrain` |
| App. F.3, Fig. 8 | Online policy transfer | Two Rooms | One-hot states | DDPG, DDPG + pretrained policy | `configs/scripts/appendix/two_rooms_finetune_ddpg.yaml` | `python train.py --config-name scripts/appendix/two_rooms_finetune_ddpg p_path=/path/to/snapshot.pt` |
| App. F.4-F.6, Fig. 9-10, Table 1 | Reward-free pretraining for transfer/scaling | Multi-Rooms | One-hot states | ROVER and baselines | `configs/scripts/appendix/multirooms_pretrain.yaml` | `python pretrain.py --config-name scripts/appendix/multirooms_pretrain` |
| App. F.4-F.5, Fig. 9-10 | Online DDPG transfer | Multi-Rooms | One-hot states | DDPG, DDPG + pretrained policy | `configs/scripts/appendix/multirooms_finetune_ddpg.yaml` | `python train.py --config-name scripts/appendix/multirooms_finetune_ddpg p_path=/path/to/snapshot.pt` |
| App. F.4, Fig. 9 | Online SAC transfer | Multi-Rooms | 84x84 RGB | SAC, SAC + pretrained policy | `configs/scripts/appendix/multirooms_finetune_sac.yaml` | `python train.py --config-name scripts/appendix/multirooms_finetune_sac p_path=/path/to/snapshot.pt` |
| App. F.7, Table 2 | Offline replay-buffer transfer | Two Rooms | One-hot states | CQL | `configs/scripts/appendix/offline_two_rooms_cql.yaml` | `python train.py --config-name scripts/appendix/offline_two_rooms_cql` |
| App. F.7, Table 2 | Offline replay-buffer transfer | Multi-Rooms | One-hot states | CQL | `configs/scripts/appendix/offline_multirooms_cql.yaml` | `python train.py --config-name scripts/appendix/offline_multirooms_cql` |
| App. F.7, Table 2 | Offline replay-buffer transfer | Two Rooms | One-hot states | Offline DDPG reference | `configs/scripts/appendix/offline_two_rooms_ddpg.yaml` | `python train.py --config-name scripts/appendix/offline_two_rooms_ddpg` |
| App. F.7, Table 2 | Offline replay-buffer transfer | Multi-Rooms | One-hot states | Offline DDPG reference | `configs/scripts/appendix/offline_multirooms_ddpg.yaml` | `python train.py --config-name scripts/appendix/offline_multirooms_ddpg` |

The default pretraining configs use `agent=rover_paper`, a paper-tuned ROVER
agent config. Baselines are selected with Hydra overrides. For example:

```bash
python pretrain.py --config-name scripts/main_text/middle_room_behavior agent=rnd_discrete seed=1
python pretrain.py --config-name scripts/main_text/middle_room_behavior agent=smm_discrete seed=1
python pretrain.py --config-name scripts/main_text/middle_room_behavior agent=icm_apt_discrete seed=1
python pretrain.py --config-name scripts/main_text/middle_room_behavior agent=maxent_discrete seed=1
python pretrain.py --config-name scripts/main_text/middle_room_behavior agent=cic_discrete seed=1
```

Goal locations can be changed by selecting a different environment config:

```bash
python train.py --config-name scripts/appendix/multirooms_finetune_ddpg \
  env=gridworld/multiplerooms5_4x4_1 \
  p_path=/path/to/snapshot.pt \
  seed=1
```

Useful older Two Rooms and Multi-Rooms appendix variants are preserved under
`configs/env/gridworld/appendix/`.

## Main Algorithm Hyperparameters

The table summarizes the principal algorithm settings encoded by the provided
YAML configs and agent configs. Full details remain in `configs/agent/` and
`configs/scripts/`; ROVER paper runs use `configs/agent/rover_paper.yaml`.

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

Hydra writes run outputs under `exp_local/` for single runs and `exp_sweep/`
for multirun sweeps. Pretraining checkpoints are written under
`models/${obs_type}/${domain}/${agent.name}/${seed}`. Replay buffers are saved
when `save_buffer=true`, which is enabled for the paper pretraining configs.

WandB logging is disabled by default in the reproducibility YAMLs. Enable it
with `use_wandb=true` and set project/entity overrides as needed.

## Quick Config Checks

Use Hydra's config-print mode to check that configs compose without launching
training:

```bash
python pretrain.py --config-name scripts/main_text/middle_room_behavior --cfg job

python train.py --config-name scripts/appendix/two_rooms_finetune_ddpg --cfg job
```
