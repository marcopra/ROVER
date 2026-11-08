#!/usr/bin/env bash
# Run a gridworld experiment config (standard ROVER or any baseline) for several seeds on one GPU.
#
#   scripts/run_gridworld.sh <config> <gpu> <seeds> [hydra overrides...]
#
# Examples:
#   scripts/run_gridworld.sh main_text/multirooms_coverage_discrete 0 "1 2 3 4 5"
#   scripts/run_gridworld.sh main_text/middle_room_behavior 1 "1" agent=rnd_discrete
#
# Outputs: $RESULTS_DIR/<config>/<agent>_seed<k>/ (default RESULTS_DIR=results), including the
# replay buffer (buffer/), snapshots (models/) and pretrain logs. Finished runs (DONE marker) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"; GPU="$2"; SEEDS="$3"; shift 3
RESULTS_DIR="${RESULTS_DIR:-results}"
AGENT_TAG="rover"; for arg in "$@"; do [[ "$arg" == agent=* ]] && AGENT_TAG="${arg#agent=}"; done
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL="${MUJOCO_GL:-egl}"
for seed in $SEEDS; do
  out="$RESULTS_DIR/$CONFIG/${AGENT_TAG}_seed${seed}"
  [[ -f "$out/DONE" ]] && { echo "skip $out"; continue; }
  mkdir -p "$out"
  python pretrain.py --config-name "scripts/$CONFIG" seed="$seed" device=cuda:0 "hydra.run.dir=$out" "$@" \
    > "$out.log" 2>&1
  touch "$out/DONE"
done
