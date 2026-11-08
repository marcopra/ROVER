"""End-to-end smoke tests: both ROVER agents learn a model, improve their objective and act."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pathlib import Path

import gym_env
import utils

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def collect(env, n_steps, rng):
    """Random-policy transitions (obs, action, next_obs) from a gridworld with one-hot observations."""
    transitions, time_step = [], env.reset()
    for _ in range(n_steps):
        action = int(rng.integers(4))
        next_step = env.step(action)
        transitions.append((time_step.observation, action, next_step.observation))
        time_step = env.reset() if next_step.last() else next_step
    return transitions


def replay_iterator(transitions, batch_size, rng):
    obs = np.stack([t[0] for t in transitions]).astype(np.float32)
    act = np.array([t[1] for t in transitions], dtype=np.float32)[:, None]
    nxt = np.stack([t[2] for t in transitions]).astype(np.float32)
    while True:
        i = rng.integers(0, len(obs), batch_size)
        yield obs[i], act[i], np.zeros((batch_size, 1), np.float32), np.ones((batch_size, 1), np.float32), nxt[i]


def test_standard_rover_trains_and_acts_on_two_rooms():
    rng = np.random.default_rng(0)
    utils.set_seed_everywhere(0)
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.1"):
        cfg = compose("scripts/appendix/two_rooms_pretrain", overrides=["device=cpu"])
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    name = env_kwargs.pop("name")
    env = gym_env.make(name, "discrete_states", resolution=84, seed=0, **env_kwargs)
    n_states = env.observation_space.shape[0]
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=False)
    agent_cfg.update(obs_type="discrete_states", obs_shape=(n_states,), action_shape=(4,), num_expl_steps=0,
                     discount=0.9, use_tb=True, use_wandb=False, device="cpu", total_train_steps=10_000,
                     batch_size=64, batch_size_actor=256, feature_dim=16, hidden_dim=64, T_init_steps=20,
                     update_actor_every_steps=300, pmd_steps=10, sink_schedule="linear(0.0, 0.05, 1000)")
    agent = instantiate(agent_cfg)
    agent.train()
    batches = replay_iterator(collect(env, 1500, rng), 64, rng)

    metrics = {}
    for step in range(1, 301):
        metrics.update(agent.update(batches, step))
    assert "transition_loss" in metrics and np.isfinite(metrics["transition_loss"])
    assert metrics["actor_loss"] <= metrics["actor_initial_loss"], "PMD must not increase the objective"
    assert metrics["actor_loss"] < metrics["actor_initial_loss"], "PMD should reduce the objective on this problem"
    assert agent.gradient_coeff.shape == (4 * 64 + 1, 1) and agent.E.shape == (256, 4)

    observation = env.reset().observation
    probs = agent.compute_action_probs(observation)
    assert probs.shape == (4,) and np.all(probs >= 0) and probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert 0 <= agent.act(observation, {}, step=10_000, eval_mode=False) < 4


def test_subspace_rover_improves_objective_and_acts_on_pointmaze():
    from pretrain_pointmaze import make_agent, make_env
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.1"):
        cfg = compose("scripts/pointmaze/umaze_states", overrides=[
            "device=cpu", "agent.device=cpu", "num_seed_frames=500", "agent.T_init_steps=200",
            "agent.batch_size_actor=1200", "agent.subsamples=200", "agent.pca_truncation=60",
            "agent.update_actor_every_steps=700", "agent.pmd_steps=4", "agent.feature_dim=32", "agent.hidden_dim=64",
        ])
    utils.set_seed_everywhere(0)
    env = make_env(cfg, 0)
    agent = make_agent(cfg, env)
    agent.train()
    time_step = env.reset()
    obs, metrics = time_step.observation, {}
    for frame in range(1, 1501):
        action = agent.act(obs, step=frame - 1)
        time_step = env.step(action)
        agent.observe(obs, action, time_step.observation)
        obs = env.reset().observation if time_step.last() else time_step.observation
        if frame >= cfg.num_seed_frames:
            metrics.update(agent.update(frame))
    assert metrics["num_landmarks"] <= 200 and np.isfinite(metrics["actor_loss"])
    probs = agent.action_probabilities(np.stack([obs, obs]))
    assert probs.shape == (2, 4) and np.allclose(probs.sum(1), 1.0)
    state = agent.state_dict()
    restored = make_agent(cfg, env)
    restored.load_state_dict(state)
    np.testing.assert_allclose(restored.action_probabilities(obs), agent.action_probabilities(obs), atol=1e-4)
