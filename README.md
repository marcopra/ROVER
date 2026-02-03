
# ROVER

This repository contains the code to reproduce all **pretraining** and **finetuning** experiments for **ROVER** and baseline methods.  
Experiments are configured using **Hydra** and can be run with the commands below.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate rover
```

All experiments were tested on **Python 3.10** and **CUDA-enabled GPUs**.

---

## Pretraining

### ROVER

We provide ROVER pretraining for both **discrete-state** and **pixel-based** observations.

#### Discrete states
```bash
python pretrain.py configs/pretrain=rover/discrete/multiplerooms
python pretrain.py configs/pretrain=rover/discrete/two_rooms
```

#### Pixels

```bash
python pretrain.py configs/pretrain=rover/pixels/multiplerooms
python pretrain.py configs/pretrain=rover/pixels/two_rooms
```

---

### Baselines

We include RND and SMM baselines for both observation modalities.
The environment can be selected via `configs/env`.

```bash
python pretrain.py configs/pretrain=rnd/discrete configs/env=<env_name>
python pretrain.py configs/pretrain=rnd/pixels configs/env=<env_name>

python pretrain.py configs/pretrain=smm/discrete configs/env=<env_name>
python pretrain.py configs/pretrain=smm/pixels configs/env=<env_name>
```

Replace `<env_name>` with any supported environment (e.g., `two_rooms7_0`).

---

## Finetuning

Finetuning is performed by initializing a downstream RL agent from a **pretrained ROVER checkpoint**.

```bash
# ROVER

python train.py \
  agent=<ddpg|sac>_discrete_with_kernel_actor \
  obs_type=<pixels|discrete_states> \
  configs/env=<env_name> \
  p_path=<path_to_pretrained_checkpoint>

# Baselines

python train.py \
  agent=<ddpg|sac|rnd_discrete|smm_discrete> \
  obs_type=<pixels|discrete_states> \
  configs/env=<env_name> \
  p_path=<path_to_pretrained_checkpoint>

```

---

### Notes

* All hyperparameters are defined in Hydra configs or explicitly on the command line.
* To reproduce multi-seed results, rerun the same command with different `seed` values.
* WandB logging can be disabled by setting `use_wandb=false`.


