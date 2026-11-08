#!/usr/bin/env bash
# Subspace-ROVER PointMaze U-Maze pretraining (paper Sec. 4.3, Table 1, state observations).
#
#   scripts/run_pointmaze_umaze.sh [GPU] [SEEDS...]
#
# Trains one run per (start position, seed) -- the paper uses 3 starts x seeds 1 2 3 -- and then
# evaluates every final checkpoint with 3 evaluation seeds x 50 trajectories.  Runs are executed
# sequentially on one GPU; outputs go to $RESULTS_DIR/pointmaze/umaze_states/start_<x>_<y>/seed<k>
# (RESULTS_DIR defaults to ./results).  Set PYTHON to select the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"; shift || true
SEEDS="${*:-1 2 3}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL="${MUJOCO_GL:-egl}"

for start in "-1.0 1.0" "1.0 1.0" "1.0 0.0"; do
  read -r x y <<< "$start"
  for seed in $SEEDS; do
    out="${RESULTS_DIR:-results}/pointmaze/umaze_states/start_${x}_${y}/seed${seed}"
    if [[ -f "$out/eval.json" ]]; then echo "skip $out (done)"; continue; fi
    mkdir -p "$out"
    "$PYTHON" pretrain_pointmaze.py --config-name scripts/pointmaze/umaze_states \
      seed="$seed" "env.pointmaze.start_position=[$x,$y]" device=cuda:0 "output_dir=$out" \
      > "$out/train.log" 2>&1 || { echo "FAILED: $out"; exit 1; }
    "$PYTHON" evaluate_pointmaze.py "$out" --eval-seeds 3 --num-trajectories 50
  done
done
