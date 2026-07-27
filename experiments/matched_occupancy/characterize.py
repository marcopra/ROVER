from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

import gym_env
from replay_buffer import load_episode

from .core import coverage_metrics, discounted_occupancy, state_ids


def load_buffer_ids(pattern: str) -> np.ndarray:
    files = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No replay episodes match {pattern}")
    return np.concatenate([state_ids(load_episode(p)["observation"]) for p in files])


def load_agent(path: str, device: str):
    payload = torch.load(path, weights_only=False, map_location=device)
    if "agent" not in payload:
        raise ValueError(f"{path} has no agent payload")
    agent = payload["agent"]
    if hasattr(agent, "device"):
        agent.device = torch.device(device)
    if hasattr(agent, "train"):
        agent.train(False)
    return agent


def make_env(cfg: dict, seed: int):
    kwargs = dict(cfg["env"])
    task = kwargs.pop("name")
    kwargs.pop("synthetic_first_transition", None)
    return gym_env.make(
        task, "discrete_states", frame_stack=1, action_repeat=1, seed=seed,
        resolution=84, grayscale=False, url=True, **kwargs
    )


def rollout(agent, env, episodes: int, horizon: int, seed: int) -> list[list[int]]:
    trajectories = []
    for episode in range(episodes):
        ts = env.reset(seed=seed + episode)
        meta, states = agent.init_meta(), [int(np.argmax(ts.observation))]
        for _ in range(horizon):
            with torch.no_grad():
                action = agent.act(ts.observation, meta, 10**9, eval_mode=False)
            ts = env.step(action)
            states.append(int(np.argmax(ts.observation)))
            if ts.last():
                break
        trajectories.append(states)
    return trajectories


def characterize(candidate: dict, cfg: dict) -> dict:
    env = make_env(cfg, cfg["occupancy"]["seed"])
    n_states = int(env.unwrapped.n_states)
    buffer_ids = load_buffer_ids(candidate["buffer_glob"])
    agent = load_agent(candidate["checkpoint"], cfg["device"])
    trajectories = rollout(
        agent, env, cfg["occupancy"]["episodes"], cfg["occupancy"]["horizon"],
        cfg["occupancy"]["seed"],
    )
    final = discounted_occupancy(trajectories, n_states, cfg["occupancy"]["gamma"])
    return {
        **candidate,
        "n_states": n_states,
        "buffer_size": int(len(buffer_ids)),
        "buffer": coverage_metrics(buffer_ids, n_states),
        "final_policy": final,
    }

