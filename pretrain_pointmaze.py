"""Reward-free pretraining of subspace ROVER on PointMaze (paper Sec. 4.3, Table 1).

    python pretrain_pointmaze.py --config-name scripts/pointmaze/umaze_states seed=1 \
        'env.pointmaze.start_position=[1.0,0.0]' device=cuda:0

Writes ``metrics.csv`` (coverage and agent diagnostics), ``final_snapshot.pt`` and
``results.json`` into ``output_dir``.
"""

import csv
import json
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import gym_env
import utils
from agent.rover_subspace import SubspaceRoverAgent
from agent.rover_visualization.domains import (
    extract_eval_trajectory_point,
    pointmaze_evaluation_seed,
    pointmaze_free_space_coverage,
)


def make_env(cfg, seed):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name")
    return gym_env.make(cfg.task_name, cfg.obs_type, frame_stack=cfg.frame_stack, action_repeat=cfg.action_repeat,
                        seed=seed, resolution=cfg.resolution, grayscale=cfg.grayscale, **env_kwargs)


def make_agent(cfg, env) -> SubspaceRoverAgent:
    obs_dim = int(np.prod(gym_env.observation_spec(env).shape))
    n_actions = int(gym_env.action_spec(env).num_values)
    agent_kwargs = OmegaConf.to_container(cfg.agent, resolve=True)
    return SubspaceRoverAgent(obs_dim=obs_dim, n_actions=n_actions, num_expl_steps=cfg.num_seed_frames,
                              discount=cfg.discount, **agent_kwargs)


def evaluate_coverage(agent, env, training_seed, frame, num_trajectories, grid_size, radius, eval_seed=0):
    """Percentage of free PointMaze grid cells visited by sampled trajectories of the stochastic policy.

    The behaviour policy is evaluated exactly as it explores (epsilon-greedy over the PMD policy), so
    the number reflects the policy that generated the replay data.  Global RNG state is restored.
    """
    rng_state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    utils.set_seed_everywhere(pointmaze_evaluation_seed(training_seed + 1000 * eval_seed, frame))
    trajectories = []
    try:
        for episode in range(num_trajectories):
            time_step = env.reset(seed=pointmaze_evaluation_seed(training_seed + 1000 * eval_seed, frame, episode))
            trajectory = [extract_eval_trajectory_point(env, time_step)]
            while not time_step.last():
                action = agent.act(time_step.observation, step=max(frame, agent.num_expl_steps))
                time_step = env.step(action)
                trajectory.append(extract_eval_trajectory_point(env, time_step))
            trajectories.append(trajectory)
    finally:
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.random.set_rng_state(rng_state[2])
    covered, free, percent = pointmaze_free_space_coverage(env, trajectories, grid_size=grid_size, radius=radius)
    return percent


@hydra.main(config_path="configs", config_name="scripts/pointmaze/umaze_states", version_base="1.1")
def main(cfg):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    utils.set_seed_everywhere(cfg.seed)
    train_env, eval_env = make_env(cfg, cfg.seed), make_env(cfg, cfg.seed)
    agent = make_agent(cfg, train_env)
    agent.train()

    log_file = (out / "metrics.csv").open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["frame", "key", "value"])
    history, best, start = [], (-1.0, 0), time.time()

    time_step = train_env.reset()
    obs = time_step.observation
    for frame in range(1, cfg.num_train_frames + 1):
        action = agent.act(obs, step=frame - 1)
        time_step = train_env.step(action)
        agent.observe(obs, action, time_step.observation)
        obs = train_env.reset().observation if time_step.last() else time_step.observation

        if frame >= cfg.num_seed_frames:
            for key, value in agent.update(frame).items():
                writer.writerow([frame, key, value])
        if frame % cfg.eval.every_frames == 0 or frame == cfg.num_train_frames:
            coverage = evaluate_coverage(agent, eval_env, cfg.seed, frame, cfg.eval.num_trajectories,
                                         cfg.eval.grid_size, cfg.eval.radius)
            history.append((frame, coverage))
            writer.writerow([frame, "coverage_pct", coverage])
            log_file.flush()
            best = max(best, (coverage, frame))
            print(f"[{time.time() - start:7.0f}s] frame {frame:>8d}  coverage {coverage:6.2f}%", flush=True)

    torch.save({"agent": agent.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, out / "final_snapshot.pt")
    (out / "results.json").write_text(json.dumps({
        "final_coverage_pct": history[-1][1], "best_coverage_pct": best[0], "best_frame": best[1],
        "history": history, "seed": cfg.seed, "start_position": list(cfg.env.pointmaze.start_position),
        "wall_clock_seconds": time.time() - start}, indent=2))
    log_file.close()


if __name__ == "__main__":
    main()
