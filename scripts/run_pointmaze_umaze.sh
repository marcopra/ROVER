#!/usr/bin/env bash
# Subspace-ROVER PointMaze U-Maze pretraining (paper Sec. 4.3, Table 1, state observations).
#
#   scripts/run_pointmaze_umaze.sh [GPU] [SEEDS...]
#
# Trains one run per (start position, seed) -- the paper uses 3 starts x seeds 1 2 3 -- and then
# evaluates every final checkpoint with 3 evaluation seeds x 50 trajectories.  Runs are executed
# sequentially on one GPU; outputs go to results/pointmaze/umaze_states/start[x, y]/seed<k>/.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"; shift || true
SEEDS=("${@:-1 2 3}")
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL="${MUJOCO_GL:-egl}"

for start in "[-1.0,1.0]" "[1.0,1.0]" "[1.0,0.0]"; do
  for seed in ${SEEDS[@]}; do
    out="results/pointmaze/umaze_states/start${start}/seed${seed}"
    if [[ -f "$out/results.json" ]]; then echo "skip $out (done)"; continue; fi
    python pretrain_pointmaze.py --config-name scripts/pointmaze/umaze_states \
      seed="$seed" "env.pointmaze.start_position=$start" device=cuda:0 \
      "output_dir=$out" > "$out.log" 2>&1 || { echo "FAILED: $out"; exit 1; }
    python evaluate_pointmaze.py "$out" --eval-seeds 3 --num-trajectories 50
  done
done
