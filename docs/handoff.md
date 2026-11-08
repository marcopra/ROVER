# Handoff notes (working file)

Started 2026-11-08 13:19 CET. Deadline 2026-11-09 03:00 CET.
Python env: `~/miniconda3/envs/dist_matching/bin/python` (torch 2.10 cu130, RTX 5090, hydra 1.3, gymnasium-robotics, mujoco, minigrid, pytest).
Research repo (read-only): `~/distribution_matching`. Beads (stealth mode) in `~/ROVER/.beads`.

## Findings from Phase 1 (research repo vs official repo)

Two ROVER code paths produced the paper's results:

1. **Dense (standard) ROVER** — gridworld experiments (Middle Room, Multi-Rooms, Maze, Two Rooms; one-hot and pixel obs).
   Research: `agent/rover.py` + `agent/rover_matchers.py` (`DistributionMatcher`) + `agent/rover_sink.py`.
   Launchers: `launchers/slurm/pretraining/gridworld/{middle_room,maze,fast_coverage}`, `.../rover/multiplerooms`.
   Official repo `agent/rover.py` is an older dense variant with explicit feature matrices (Φ̃ of size (n+1)x(d+1)).
2. **Subspace (Nyström) ROVER** — PointMaze UMaze (Table 1), states + pixels.
   Research: `agent/rover_nystrom_subspace.py` (`RoverSubspaceCoverageAgent`) on top of `agent/rover_nystrom_debug.py`
   + `agent/rover_matchers.py::NystromActorModel` + `agent/rover_buffers.py` (pivoted Cholesky landmark selection)
   + `agent/rover_subspace_matchers.py` + `agent/rover_sink.py`. Entry: `pretrain_parallel.py --config-name=pretrain_parallel/pointmaze_subspace_{states,pixels}`.
   Paper runs: launchers `launchers/slurm/pretraining/pointmaze_subspace/umaze/{states,pixels}/rover/{launcher,base}.sh`
   (3 start positions x 3 seeds, 250k frames). Existing pixel logs: `exp_benchmark/exp_local_parallel2/2026.10.2{3,4}/*`.

Key research-only fixes found in git history:
- `42c3e3c` (2026-05-28) gradient: kernel-based gradient (Gram of sink-augmented features, state-action kernel K with action mask,
  transpose fix `(I-γBM)^T A^T` ordering / right term `(I-γBM)^{-1}`).
- `a007425` (2026-10-05) subspace: coverage kernel on projection g(s)=(x,y), dynamics kernel on full state.
- `98142db` sink state w/ Gaussian kernel: sink Gram must be built from kernel Gram + sink terms (`orthogonal_sink_residual_gram`),
  not a kernel applied to concatenated features.
- `eb935af` (2026-10-12) goal leakage: `render_image_observation()` / `render()` default showed the goal (rooms, continuous rooms
  text overlay of goal coords, classic minigrid); PointMaze state obs concatenated dict keys incl. `desired_goal`
  → `PointMazeAgentStateObservationWrapper` (`only_agent_state`). Also `continuing_task` explicit for PointMaze.
- `bddafb1` (2026-10-22) pivoted Cholesky landmark selection on device (one host sync per pivot, per-action blocks).
- `d062598` memory efficiency; `950c906` blockwise regression (NystromActorModel: per-action LU blocks, top-r eigenspace).

(see docs/implementation_summary.md once written for the final version)
