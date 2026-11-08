"""Offline RL (CQL / DDPG) on a replay buffer collected during reward-free pretraining (paper App. G.7, Table 3).

    python train_offline.py --config-name scripts/appendix/offline_two_rooms_cql \
        replay_buffer_dir=/path/to/pretraining/run/buffer seed=1

The buffer is the ``buffer/`` directory written by ``pretrain.py`` (``save_buffer=true``); rewards are
relabelled with the downstream task (goal given by the env config).  The policy is evaluated
deterministically every ``eval_every_steps`` gradient steps and after the last one.
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
from dm_env import specs
from omegaconf import OmegaConf

import gym_env
import utils
from logger import Logger
from replay_buffer import make_offline_replay_loader


def make_agent(obs_type, obs_spec, action_spec, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape if obs_spec.shape else (1,)
    cfg.action_shape = (action_spec.num_values,) if hasattr(action_spec, "num_values") else action_spec.shape
    cfg.num_expl_steps = 0
    return hydra.utils.instantiate(cfg)


def evaluate(global_step, agent, env, logger, num_episodes):
    """Mean undiscounted return of the deterministic policy."""
    total_reward, steps = 0.0, 0
    meta = agent.init_meta()
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, meta, global_step, eval_mode=True)
            time_step = env.step(action)
            total_reward += time_step.reward
            steps += 1
    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("episode_reward", total_reward / num_episodes)
        log("episode_length", steps / num_episodes)
        log("step", global_step)
    return total_reward / num_episodes


@hydra.main(config_path="configs", config_name="scripts/appendix/offline_two_rooms_cql", version_base="1.1")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")
    utils.set_seed_everywhere(cfg.seed)
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    env = gym_env.make(cfg.task_name, cfg.obs_type, frame_stack=cfg.frame_stack, action_repeat=cfg.action_repeat,
                       seed=cfg.seed, resolution=cfg.resolution, grayscale=cfg.grayscale, url=False, **env_kwargs)
    obs_spec, action_spec = gym_env.observation_spec(env), gym_env.action_spec(env)
    agent = make_agent(cfg.obs_type, obs_spec, action_spec, cfg.agent)

    replay_dir = Path(cfg.replay_buffer_dir).resolve()
    assert replay_dir.is_dir() and any(replay_dir.glob("*.npz")), f"no episodes in {replay_dir}"
    print(f"replay dir: {replay_dir}")
    replay_loader = make_offline_replay_loader(env, replay_dir, cfg.replay_buffer_size, cfg.batch_size,
                                               cfg.replay_buffer_num_workers, cfg.discount,
                                               relable=bool(cfg.relable))
    replay_iter = iter(replay_loader)

    eval_every = utils.Every(cfg.eval_every_steps)
    log_every = utils.Every(cfg.log_every_steps)
    timer = utils.Timer()
    for global_step in range(cfg.num_grad_steps):
        if eval_every(global_step):
            logger.log("eval_total_time", timer.total_time(), global_step)
            evaluate(global_step, agent, env, logger, cfg.num_eval_episodes)
        logger.log_metrics(agent.update(replay_iter, global_step), global_step, ty="train")
        if log_every(global_step):
            elapsed, total = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed)
                log("total_time", total)
                log("step", global_step)
    final = evaluate(cfg.num_grad_steps, agent, env, logger, cfg.num_eval_episodes)
    (work_dir / "final_return.txt").write_text(f"{final}\n")
    print(f"final evaluation return after {cfg.num_grad_steps} gradient steps: {final}")


if __name__ == "__main__":
    main()
