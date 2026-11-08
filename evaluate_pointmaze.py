"""Evaluate a pretrained subspace-ROVER PointMaze checkpoint (paper Table 1 protocol).

    python evaluate_pointmaze.py results/pointmaze/umaze_states/start[-1.0,1.0]/seed1

Coverage is the percentage of free grid cells (90 x 90 grid, radius 0.08) visited by 50 trajectories
(300 steps) of the stochastic behaviour policy, averaged over ``--eval-seeds`` evaluation seeds.
Results are written to ``eval.json`` next to the checkpoint.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from pretrain_pointmaze import evaluate_coverage, make_agent, make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshot", default="final_snapshot.pt")
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--device", default=None, help="override the device stored in the run config")
    args = parser.parse_args()

    snapshot = torch.load(args.run_dir / args.snapshot, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(snapshot["cfg"])
    if args.device is not None:
        cfg.device, cfg.agent.device = args.device, args.device
    env = make_env(cfg, cfg.seed)
    agent = make_agent(cfg, env)
    agent.load_state_dict(snapshot["agent"])
    agent.train(False)
    values = [evaluate_coverage(agent, env, cfg.seed, cfg.num_train_frames, args.num_trajectories,
                                cfg.eval.grid_size, cfg.eval.radius, eval_seed=k) for k in range(args.eval_seeds)]
    result = {"snapshot": args.snapshot, "coverage_pct_per_eval_seed": values, "coverage_pct_mean": float(np.mean(values)),
              "num_trajectories": args.num_trajectories, "training_seed": cfg.seed,
              "start_position": list(cfg.env.pointmaze.start_position)}
    (args.run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
