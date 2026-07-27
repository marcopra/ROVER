# Experiment 1: matched buffer coverage vs final occupancy

Status: implementation complete; full GPU experiment not run in this checkout. No
scientific claim is made until `results.json` exists and confidence intervals support it.

## Existing pipelines reused

- Reward-free pretraining: `pretrain.py`, including saved snapshots and replay episodes.
- Checkpoint evaluation: `Workspace.eval()` plus direct checkpoint `agent.act()` rollouts.
- Replay analysis: compressed episode files written by `ReplayBufferStorage`.
- Sparse-reward adaptation: `train.py` and the appendix MultiRooms DDPG transfer setup.

MultiRooms uses fixed one-hot states. This isolates policy transfer from representation
quality while keeping architecture, encoder-transfer behavior, optimizer, horizon,
environment steps, and adaptation budget identical within each pair.

## Protocol

Candidate selection uses only pretraining outputs. Raw replay support must differ by at
most 5% (2% target), while normalized final discounted effective-support gap must be at
least 10%. Ranking maximizes that gap, then final-occupancy Jensen-Shannon divergence,
then minimizes buffer-support gap. Adaptation results are rejected as matching inputs.
If no pair exists, status becomes `no_eligible_pairs`; use deterministic equal-size
buffer subsampling declared in config, never policy modification.

Goals are room centers in near (room 1), intermediate (room 3), and distant (room 5)
regions. Every checkpoint uses seeds 1–5. Occupancy uses 100 stochastic rollouts,
horizon 500, gamma 0.99, and a fixed independent seed.

## Commands

```bash
conda activate dist_matching

# Pretrain ROVER and CIC for every pretraining seed/checkpoint requested in candidate list.
python pretrain.py --config-name scripts/appendix/multirooms_pretrain -m agent=rover_paper seed=1,2,3
python pretrain.py --config-name scripts/appendix/multirooms_pretrain -m agent=cic_discrete seed=1,2,3

# Edit exact checkpoint and replay globs in config, then characterize and freeze pairs.
python -m experiments.matched_occupancy.cli characterize \
  --config configs/experiments/matched_occupancy.yaml \
  --output results/experiment1/characterization.json
python -m experiments.matched_occupancy.cli match \
  --config configs/experiments/matched_occupancy.yaml \
  --input results/experiment1/characterization.json \
  --output results/experiment1/frozen_pairs.json

# Generate commands only after pair file is frozen. Inspect, then run.
python -m experiments.matched_occupancy.launch \
  --config configs/experiments/matched_occupancy.yaml \
  --pairs results/experiment1/frozen_pairs.json \
  --characterization results/experiment1/characterization.json
bash results/experiment1/launch_commands.sh

python -m experiments.matched_occupancy.analyze \
  --config configs/experiments/matched_occupancy.yaml \
  --pairs results/experiment1/frozen_pairs.json \
  --characterization results/experiment1/characterization.json

python -m unittest tests.test_matched_occupancy
```

Outputs: frozen pair JSON, JSONL run records, machine-readable `results.json`, CSV
matched-pair tables, buffer/final-occupancy scatterplot, and per-goal learning curves.
