"""PointMaze evaluation utilities: trajectory points and the free-space coverage metric.

Coverage (paper Sec. 4.3, App. G.8) is the percentage of free cells of a regular grid over the maze
(wall cells removed) that lie within ``radius`` of at least one sampled trajectory point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _find_discrete_env(reference):
    current = reference
    while current is not None:
        if all(hasattr(current, attr) for attr in ("n_states", "idx_to_state", "state_to_idx")):
            return current

        if hasattr(current, "env"):
            current = current.env
        elif hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        else:
            break
    return None


def _get_env_method(env, method_name: str):
    current = env
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)

    return None


def _as_xy(value) -> Optional[np.ndarray]:
    if value is None:
        return None

    xy = np.asarray(value, dtype=np.float32).reshape(-1)
    if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
        return None
    return xy[:2]


def extract_eval_trajectory_point(env, time_step) -> Optional[np.ndarray]:
    """Extract an x/y point from an evaluation time step when available."""
    info = getattr(time_step, "info", None)
    if isinstance(info, dict):
        for key in ("agent_position", "position", "xy"):
            xy = _as_xy(info.get(key))
            if xy is not None:
                return xy

    method = _get_env_method(env, "get_debug_coordinates")
    if callable(method):
        debug_info = method()
        if isinstance(debug_info, dict):
            for key in ("xy", "xyz", "agent_position", "position"):
                xy = _as_xy(debug_info.get(key))
                if xy is not None:
                    return xy

    discrete_env = _find_discrete_env(env)
    raw_proprio = getattr(time_step, "proprio_observation", [])
    proprio_array = np.asarray(raw_proprio, dtype=np.float32)
    proprio = proprio_array.reshape(-1)
    if discrete_env is not None and proprio.size == getattr(discrete_env, "n_states", -1):
        state_idx = int(np.argmax(proprio))
        return _as_xy(discrete_env.idx_to_state.get(state_idx))

    if proprio_array.ndim <= 1:
        return _as_xy(proprio)
    return None


def _prepare_trajectories(trajectories) -> list[np.ndarray]:
    prepared = []
    for trajectory in trajectories:
        if trajectory is None:
            continue
        points = [_as_xy(point) for point in trajectory]
        points = [point for point in points if point is not None]
        if points:
            prepared.append(np.asarray(points, dtype=np.float32))
    return prepared


def pointmaze_free_space_coverage(
    env,
    trajectories,
    grid_size: int = 90,
    radius: float = 0.08,
) -> tuple[int, int, float]:
    """Measure trajectory coverage over a regular grid of PointMaze free space."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if radius <= 0:
        raise ValueError("radius must be positive")

    prepared = _prepare_trajectories(trajectories)
    if not prepared:
        raise ValueError("coverage requires at least one valid trajectory point")

    layout = _get_pointmaze_wall_layout(env)
    if layout is None:
        raise AttributeError("PointMaze environment has no valid get_debug_maze_layout()")

    lower = layout["maze_lower"]
    upper = layout["maze_upper"]
    walls = layout["wall_rectangles"]
    xs = np.linspace(lower[0], upper[0], grid_size, dtype=np.float32)
    ys = np.linspace(lower[1], upper[1], grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    inside_wall = np.zeros(len(grid), dtype=bool)
    if walls.size:
        x = grid[:, 0:1]
        y = grid[:, 1:2]
        inside_wall = (
            (x >= walls[:, 0])
            & (x <= walls[:, 0] + walls[:, 2])
            & (y >= walls[:, 1])
            & (y <= walls[:, 1] + walls[:, 3])
        ).any(axis=1)
    free = grid[~inside_wall]
    if len(free) == 0:
        raise ValueError("PointMaze layout contains no free grid points")

    samples = np.concatenate(prepared, axis=0)
    covered = np.zeros(len(free), dtype=bool)
    radius_sq = float(radius) ** 2
    for start in range(0, len(free), 2048):
        chunk = free[start : start + 2048]
        distances = ((chunk[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2)
        covered[start : start + len(chunk)] = distances.min(axis=1) <= radius_sq

    covered_count = int(covered.sum())
    free_count = int(len(free))
    return covered_count, free_count, 100.0 * covered_count / free_count


@dataclass
class CoverageProgress:
    """Track checkpoint-to-checkpoint coverage growth for logging."""

    tolerance: float = 0.25
    previous_coverage: Optional[float] = None
    previous_frame: Optional[int] = None
    best_coverage: float = 0.0

    def update(self, coverage_pct: float, frame: int) -> dict[str, float]:
        coverage_pct = float(coverage_pct)
        frame = int(frame)
        if not np.isfinite(coverage_pct):
            raise ValueError("coverage_pct must be finite")
        if frame < 0:
            raise ValueError("frame must be non-negative")
        if self.previous_frame is not None and frame <= self.previous_frame:
            raise ValueError("coverage frames must increase strictly")

        delta = 0.0 if self.previous_coverage is None else coverage_pct - self.previous_coverage
        frame_delta = 0 if self.previous_frame is None else frame - self.previous_frame
        gain_per_100k = 0.0 if frame_delta == 0 else delta * 100_000.0 / frame_delta
        self.best_coverage = max(self.best_coverage, coverage_pct)
        self.previous_coverage = coverage_pct
        self.previous_frame = frame
        return {
            "coverage_pct": coverage_pct,
            "coverage_delta": delta,
            "coverage_best": self.best_coverage,
            "coverage_gain_per_100k": gain_per_100k,
            "coverage_expanding": float(delta > self.tolerance),
        }


def pointmaze_evaluation_seed(training_seed: int, frame: int, episode: int = 0) -> int:
    """Stable per-run, per-checkpoint seed without Python hash randomization."""
    return int(training_seed) * 1_000_003 + int(frame) + int(episode)


def _get_pointmaze_wall_layout(env):
    method = _get_env_method(env, "get_debug_maze_layout")
    if not callable(method):
        return None

    layout = method()
    if not isinstance(layout, dict):
        return None

    required = ("maze_lower", "maze_upper", "wall_rectangles")
    if any(key not in layout for key in required):
        return None

    return {
        "maze_lower": np.asarray(layout["maze_lower"], dtype=np.float32),
        "maze_upper": np.asarray(layout["maze_upper"], dtype=np.float32),
        "wall_rectangles": np.asarray(layout["wall_rectangles"], dtype=np.float32).reshape(-1, 4),
    }
