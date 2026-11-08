"""Tests for the PointMaze free-space coverage metric used in Table 1."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

import gym_env
from agent.rover_visualization.domains import (
    CoverageProgress,
    extract_eval_trajectory_point,
    pointmaze_evaluation_seed,
    pointmaze_free_space_coverage,
)


@pytest.fixture(scope="module")
def env():
    kwargs = dict(render_mode="rgb_array", continuing_task=True,
                  pointmaze=dict(top_down_camera=True, discrete_actions=True, direct_velocity_actions=False,
                                 only_xy_position=False, only_agent_state=True, start_position=[-1.0, 1.0],
                                 start_position_variance=0.0, reward_type="sparse", goal_position=[1.0, 1.0]))
    return gym_env.make("PointMaze_UMaze-v3", obs_type="states", resolution=84, seed=0, grayscale=True, **kwargs)


def test_coverage_of_a_dense_cloud_is_complete_and_of_one_point_is_tiny(env):
    layout = env.get_debug_maze_layout()
    low, high = layout["maze_lower"], layout["maze_upper"]
    cloud = np.stack(np.meshgrid(np.linspace(low[0], high[0], 200), np.linspace(low[1], high[1], 200)), -1).reshape(-1, 2)
    _, free, full = pointmaze_free_space_coverage(env, [cloud], grid_size=90, radius=0.08)
    assert full == pytest.approx(100.0) and free > 0
    _, _, single = pointmaze_free_space_coverage(env, [np.array([[-1.0, 1.0]])], grid_size=90, radius=0.08)
    assert 0.0 < single < 2.0


def test_coverage_is_monotone_in_radius_and_trajectories(env):
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.5, 1.5, size=(60, 2))
    small = pointmaze_free_space_coverage(env, [points[:30]], radius=0.08)[2]
    more = pointmaze_free_space_coverage(env, [points], radius=0.08)[2]
    wider = pointmaze_free_space_coverage(env, [points], radius=0.2)[2]
    assert small <= more <= wider


def test_trajectory_point_is_the_agent_position(env):
    time_step = env.reset()
    point = extract_eval_trajectory_point(env, time_step)
    np.testing.assert_allclose(point, time_step.observation[:2], atol=1e-5)


def test_coverage_progress_and_eval_seed():
    progress = CoverageProgress(tolerance=0.25)
    first = progress.update(50.0, 1000)
    second = progress.update(60.0, 2000)
    assert first["coverage_delta"] == 0 and second["coverage_delta"] == pytest.approx(10.0)
    assert second["coverage_expanding"] == 1.0 and second["coverage_best"] == 60.0
    with pytest.raises(ValueError):
        progress.update(61.0, 2000)           # frames must increase strictly
    assert pointmaze_evaluation_seed(1, 20000, 3) != pointmaze_evaluation_seed(1, 20000, 4)
