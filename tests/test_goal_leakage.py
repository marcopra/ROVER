"""Regression tests: the goal must never be visible to the agent's observations.

Historical bug (research repo commit eb935af): ``render_image_observation()`` and the
default ``render()`` of the room environments drew the goal marker, continuous rooms
printed the goal coordinates on the frame, and PointMaze state observations were built by
concatenating every key of the dict observation, including ``desired_goal``.  Reward-free
pretraining must be goal-agnostic, so every observation modality must be identical for two
episodes that differ only in the goal location.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

import gym_env

ROOMS_KWARGS = dict(room_size=5, corridor_length=1, corridor_y=2, max_steps=20,
                    show_coordinates=False, start_position=[0, 0], render_mode="rgb_array")


def _first_step(goal_position, obs_type, name="TwoRooms-v0", **kwargs):
    env = gym_env.make(name, obs_type=obs_type, resolution=84, seed=0, goal_position=goal_position, **kwargs)
    time_step = env.reset()
    return env, time_step


@pytest.mark.parametrize("obs_type", ["pixels", "discrete_states"])
def test_room_observations_do_not_depend_on_goal(obs_type):
    _, ts_a = _first_step([8, 2], obs_type, **ROOMS_KWARGS)
    _, ts_b = _first_step([9, 4], obs_type, **ROOMS_KWARGS)
    np.testing.assert_array_equal(ts_a.observation, ts_b.observation)
    # The auxiliary frame (videos, debugging) is goal-free as well.
    np.testing.assert_array_equal(ts_a.image_observation, ts_b.image_observation)


def test_room_default_render_hides_goal():
    env_a, _ = _first_step([8, 2], "pixels", **ROOMS_KWARGS)
    env_b, _ = _first_step([9, 4], "pixels", **ROOMS_KWARGS)
    base_a, base_b = env_a.unwrapped, env_b.unwrapped
    np.testing.assert_array_equal(base_a.render(), base_b.render())
    np.testing.assert_array_equal(base_a.render_image_observation(), base_b.render_image_observation())
    # Sanity: the goal marker is available when explicitly requested, so the test is not vacuous.
    assert not np.array_equal(base_a.render(show_goal=True), base_b.render(show_goal=True))


POINTMAZE_KWARGS = dict(render_mode="rgb_array", continuing_task=True,
                        pointmaze=dict(top_down_camera=True, discrete_actions=True, direct_velocity_actions=False,
                                       only_xy_position=False, only_agent_state=True, start_position=[-1.0, 1.0],
                                       start_position_variance=0.0, reward_type="sparse"))


def _pointmaze_step(goal, obs_type):
    kwargs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in POINTMAZE_KWARGS.items()}
    kwargs["pointmaze"]["goal_position"] = goal
    env = gym_env.make("PointMaze_UMaze-v3", obs_type=obs_type, resolution=84, seed=0, grayscale=True, **kwargs)
    return env, env.reset()


def test_pointmaze_state_observation_is_agent_state_only():
    _, ts_a = _pointmaze_step([1.0, 1.0], "states")
    _, ts_b = _pointmaze_step([-1.0, 0.0], "states")
    assert ts_a.observation.shape == (4,), "state observation must be (x, y, vx, vy) only"
    np.testing.assert_array_equal(ts_a.observation, ts_b.observation)


def test_pointmaze_pixel_observation_does_not_depend_on_goal():
    _, ts_a = _pointmaze_step([1.0, 1.0], "pixels")
    _, ts_b = _pointmaze_step([-1.0, 0.0], "pixels")
    np.testing.assert_array_equal(ts_a.observation, ts_b.observation)
    np.testing.assert_array_equal(ts_a.image_observation, ts_b.image_observation)
