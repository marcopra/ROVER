"""Samples needed to cover a given fraction of a gridworld, from a saved replay buffer.

Used for the full-coverage sample-efficiency results (paper Sec. 4.2 / Fig. 3) and the
scaling table (App. G.6).  The replay buffer of a pretraining run (``save_buffer=true``)
is scanned in collection order; the output is the number of environment samples after
which the visited fraction of spatial cells first reaches each target.

Only discrete one-hot observations are supported (the state index is the argmax).

    python scripts/coverage_speed.py results/repro/multirooms_discrete/rover_s1
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gym_env  # noqa: E402


def spatial_cells(env):
    """Unique (x, y) cells of the environment, excluding the absorbing dead state."""
    unwrapped = env.unwrapped
    dead = getattr(unwrapped, "DEAD_STATE", None)
    dead_cell = None if dead is None else tuple(dead[:2])
    if getattr(unwrapped, "plot_cells", None):
        cells = [tuple(cell[:2]) for cell in unwrapped.plot_cells]
    elif getattr(unwrapped, "cells", None):
        cells = [tuple(cell[:2]) for cell in unwrapped.cells]
    else:
        cells = [tuple(unwrapped.idx_to_state[i][:2]) for i in range(unwrapped.n_states)]
    return list(dict.fromkeys(c for c in cells if c != dead_cell)), dead_cell


def state_to_cell_index(env):
    unwrapped = env.unwrapped
    cells, dead_cell = spatial_cells(env)
    lookup = {cell: i for i, cell in enumerate(cells)}
    mapping = np.full(unwrapped.n_states, -1, dtype=np.int64)
    for state_idx in range(unwrapped.n_states):
        cell = tuple(unwrapped.idx_to_state[state_idx][:2])
        if cell != dead_cell:
            mapping[state_idx] = lookup[cell]
    return mapping, len(cells)


def coverage_speed(episode_files, env, targets_pct):
    """First sample count at which each coverage target is reached (None if never)."""
    mapping, n_cells = state_to_cell_index(env)
    required = {pct: math.ceil(n_cells * pct / 100.0) for pct in targets_pct}
    reached = {pct: None for pct in targets_pct}
    visited = np.zeros(n_cells, dtype=bool)
    count = samples = 0
    for path in episode_files:
        with np.load(path) as episode:
            observations = episode["observation"]
        assert observations.ndim == 2, f"expected one-hot observations, got {observations.shape}"
        for cell in mapping[observations.argmax(axis=1)]:
            samples += 1
            if cell >= 0 and not visited[cell]:
                visited[cell] = True
                count += 1
                for pct in targets_pct:
                    if reached[pct] is None and count >= required[pct]:
                        reached[pct] = samples
        if all(v is not None for v in reached.values()):
            break
    return reached, count, n_cells, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="pretraining run directory (contains .hydra/ and buffer/)")
    parser.add_argument("--targets", type=float, nargs="+", default=[25, 50, 75, 90, 95, 99, 100])
    parser.add_argument("--output", type=Path, default=None, help="CSV path (default: <run_dir>/coverage_speed.csv)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra" / "config.yaml")
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    env = gym_env.make(cfg.task_name, cfg.obs_type, frame_stack=cfg.frame_stack, action_repeat=cfg.action_repeat,
                       seed=cfg.seed, resolution=cfg.resolution, grayscale=cfg.grayscale, **env_kwargs)
    files = sorted((args.run_dir / "buffer").glob("*.npz"))
    assert files, f"no episodes in {args.run_dir / 'buffer'}"
    reached, visited, n_cells, samples = coverage_speed(files, env, args.targets)

    output = args.output or args.run_dir / "coverage_speed.csv"
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_coverage_pct", "samples_to_reach"])
        for pct, count in reached.items():
            writer.writerow([pct, "" if count is None else count])
    print(f"final coverage {visited}/{n_cells} cells after {samples} samples")
    for pct, count in reached.items():
        print(f"{pct:6.1f}% -> {'not reached' if count is None else count}")


if __name__ == "__main__":
    main()
