# Implementation summary

This document records how the official ROVER release was updated from the latest research code
(`distribution_matching`, read-only reference) and what was verified. Equation and section numbers
refer to `rover.pdf`.

## 1. Which code produced which result

| Paper result | Algorithm | Research code (reference) | This repository |
| --- | --- | --- | --- |
| Gridworld experiments: Fig. 2, 3, 7-10, Tables 2-3 (Middle Room, Multi-Rooms, Maze, Two Rooms; one-hot and pixel observations) | **Standard (dense) ROVER**, inner-product kernel on L1-normalised learned features | `agent/rover.py` + `agent/rover_matchers.py::DistributionMatcher` | `agent/rover.py`, `agent/rover_math.py` |
| PointMaze U-Maze, Table 1 (states and pixels) | **Subspace ROVER** with Nystrom approximation | `agent/rover_nystrom_subspace.py` (on `rover_nystrom_debug.py`), `rover_matchers.py::NystromActorModel`, `rover_buffers.py` (pivoted Cholesky), `rover_sink.py` | `agent/rover_subspace.py`, `agent/nystrom.py`, `agent/landmarks.py`, `agent/kernels.py` |

Evidence for the attribution: the gridworld launchers (`launchers/slurm/pretraining/gridworld/*`) use
`agent=rover`; the PointMaze launchers (`launchers/slurm/pretraining/pointmaze_subspace/*`) use
`RoverSubspaceCoverageAgent`, and the W&B run manifests in `paper_ready_plots/metadata/run_manifest.csv`
store the exact per-run configuration (e.g. `umaze_states_rover_gaussian_batch100000_dim256_nys10000_sink1_...`).
Pixel PointMaze (Table 1, right column) is **not** ported (see Remaining issues).

## 2. Differences found between the official and the research implementation

Determined by diffing the two trees and walking the research git history (`git log` messages
containing "MAJOR").

| # | Difference | Research commit | Status here |
| --- | --- | --- | --- |
| 1 | **Goal leakage**: `render_image_observation()` and the default `render()` of room environments drew the goal marker; continuous rooms printed the goal coordinates on the frame; classic MiniGrid frames contained the goal tile; PointMaze state observations were built by concatenating *all* keys of the dict observation, including `desired_goal` (8-dim instead of 4-dim). | `eb935af` | Fixed, regression-tested (`tests/test_goal_leakage.py`). |
| 2 | **Subspace ROVER**: dynamics kernel on the full learned state features, coverage kernel on the projection `g(s) = (x, y)` (App. E). | `a007425` | Ported (`agent/rover_subspace.py`). |
| 3 | **Gradient with general kernels**: the research matcher builds the gradient from the *kernel* Gram of the sink-augmented features and an action-masked state-action kernel `K = k(x, x') 1[a = a']`; earlier it used `psi psi^T` and had a transposition error in the `(I - gamma A^-1 M)` solves (`42c3e3c`). The sink Gram must be assembled from the state Gram (`98142db`), because a nonlinear kernel applied to concatenated residual coordinates does not preserve the sink algebra. | `42c3e3c`, `98142db` | Ported as `sink_residual_gram` / `DenseOccupancyModel`. For the inner-product kernel the original official gradient was already correct (verified, Sec. 4). |
| 4 | **GPU pivoted Cholesky** for landmark selection: per-action Cholesky factors, everything on device, one host sync per pivot, geometric factor growth. | `c2c623f`, `bddafb1` | Ported (`agent/landmarks.py`), tested against brute force and CPU/GPU. |
| 5 | **Nystrom actor model**: per-action normal equations factorised once (LU), reduced-rank adjoint solve on the top-`rank` eigenspace of the landmark Gram, float64. | `950c906`, `d062598` | Ported (`agent/nystrom.py`), validated against the dense model and against a stored research policy (Sec. 4). |
| 6 | **Update schedule**: the official `update()` skipped the actor update whenever `step % update_every_steps != 0`; the research version decouples encoder and actor schedules. | `ae939bb` and later | Fixed. |
| 7 | **Line search**: the research version after the image-fix audit rejects non-descent steps and stops at the last accepted policy instead of accepting a worsening step after the maximum number of shrinks. | `docs/image_rover_fix_verification.md` | Adopted for both agents. |
| 8 | PointMaze environment: explicit `continuing_task`, `reward_type`, start-position variance, `__getattr__` recursion guard on wrappers. | `eb935af` and others | Ported with the environment files. |
| 9 | Pretraining PointMaze evaluation: coverage metric (`pointmaze_free_space_coverage`, 90x90 grid, radius 0.08, 50 trajectories, 300 steps), deterministic per-checkpoint evaluation seeds. | - | Ported (`agent/rover_visualization/domains.py`, `pretrain_pointmaze.py`). |

Not ported on purpose (unused in the published experiments): CURL loss, reward head, adagrad/adadiff
step-size modes, gradient clipping, PMD warm start, frozen/random-encoder variants, positive-kernel
variants, whitening, LeWorld loss, fixed-dataset debug mode, Montezuma/Atari/Fetch ROVER variants, the
multi-process parallel runner, W&B-specific hooks and the t-SNE/occupancy debug visualisers.

## 3. Changes implemented

* `agent/rover_math.py` -- dense objective and gradient (Eq. 8, Thm. 2, Prop. 4, Thm. 5). `K + lambda I` is
  Cholesky-factorised once per actor update; each resolvent matrix is LU-factorised once and shared by the
  loss and the gradient (adjoint solve), instead of four `n x n` factorisations per PMD iteration. The
  sink Gram `G` is built from the state Gram (`sink_residual_gram`); no `(n+1) x (d+1)` feature matrix is formed.
* `agent/rover.py` -- standard ROVER agent, ~300 lines (was 2711, of which ~1400 were duplicated visualiser code).
  Same encoder/InfoNCE/PMD semantics; the actor batch is `floor(batch_size_actor / batch_size) * batch_size`
  transitions (4096 for the default 5000/1024, as in the published runs). Interfaces used by transfer
  (`policy_encoder`, `_phi_all_obs`, `gradient_coeff`, `E`) are unchanged.
* `agent/rover_networks.py`, `agent/kernels.py` -- encoders, predictor, Gaussian/inner-product kernels with median bandwidth.
* `agent/landmarks.py`, `agent/nystrom.py`, `agent/rover_subspace.py` -- subspace ROVER (below).
* `env/*` -- goal-leakage fixes and PointMaze agent-state wrapper; `pretrain_pointmaze.py`, `evaluate_pointmaze.py`,
  `configs/scripts/pointmaze/umaze_states.yaml`, `scripts/` -- experiment entry points.
* `utils.schedule` -- four-argument `linear(init, final, start, end)` schedule used by the PointMaze sink schedule.

### Subspace ROVER and its relation to standard ROVER

Standard ROVER minimises `J(pi) = ||mu_{d^pi}||^2_H` with the *same* features defining the transition
operator and the kernel. Subspace ROVER keeps the operator on the full (learned) state features but measures
coverage of the pushforward `q^pi = g_# d^pi` (App. E, Eqs. 75-88):

```
J_c(pi) = (1 - gamma)^2 z^T G_c z,    G_c = sink-augmented Gram of k_c(g(s'_l), g(s'_m)) over the landmark next states
(I - gamma blockdiag(T(pi), 1)) z = [alpha; 1]
```

so only the Gram matrix `G` in Eq. (8) changes. With `g = id` and `k_c = k_d` it is exactly standard ROVER; this is
tested (`test_nystrom_with_all_landmarks_matches_dense_loss_and_gradient`: with all samples as landmarks and
full rank the Nystrom model reproduces the dense loss and gradient). On top of this, the operator is estimated
on `m << n` landmarks per action (Sec. 3.1) and the adjoint solve is restricted to the top-`r` eigenspace of the
landmark Gram (`pca_truncation`), which makes each PMD step `O(m^3)`/`O(r^3)` instead of `O(n^3)`.

## 4. Verification performed (all executed; see `tests/`)

| Check | Test | Result |
| --- | --- | --- |
| Dense loss and gradient equal the original explicit-feature implementation | `tests/test_rover_math.py::test_loss_and_gradient_match_explicit_feature_reference` | pass, 1e-10 rel. |
| Thm. 2 gradient equals autograd of `J` through the policy table (table gradient = `z_j` x functional gradient, up to a per-state constant) | `test_gradient_matches_autograd_of_the_objective` | pass, 1e-8 rel. |
| Thm. 5 decomposition `||mu||^2 = ||mu_hat||^2 + eps^2 m_e^2`; Gram PSD | `test_objective_decomposes_...`, `test_sink_gram_...` | pass |
| float32 (training dtype) vs float64 | `test_float32_gradient_is_close_to_float64` | pass (<5e-3 rel.) |
| CPU vs GPU (dense, Nystrom, pivoted Cholesky) | `test_cpu_gpu_*` | pass |
| Non-PD Gram fails loudly | `test_non_positive_definite_gram_fails_loudly` | pass |
| Nystrom (all landmarks, full rank) equals dense for Gaussian kernel | `tests/test_nystrom_subspace.py` | pass |
| Pivoted Cholesky equals brute-force greedy pivots (Gaussian/inner product, forced first) | `test_pivoted_cholesky_matches_brute_force` | pass |
| Reduced-rank gradient error decreases with the rank | `test_reduced_rank_gradient_error_decreases_with_rank` | pass |
| Goal never appears in any observation modality | `tests/test_goal_leakage.py` | pass; fails on the pre-fix environment files |
| Experiment configs compose and match constructors | `tests/test_configs.py` | pass |
| **Policy equivalence with the research code**: running the ported PMD on the stored actor batch of a research PointMaze checkpoint reproduces its policy (mean abs. diff 0.005, objective 0.5666 vs 0.5657); evaluating the research policy with the ported acting/evaluation code gives 78.3/74.7/74.3 % coverage vs. 73.5 % reported | ad-hoc scripts, results in `docs/reproduction_report.md` | consistent |

## 5. Remaining implementation issues

* Pixel PointMaze (Table 1, right column) and the pixel/Atari variants of subspace ROVER are not ported.
* Subspace ROVER uses one environment and a single process; the research runs used 4 parallel environments
  (same update schedule in frames), which changes the sample stream but not the algorithm.
* The Nystrom normal equations carry fixed 1e-6 ridge jitters (kept for fidelity); for very smooth kernels
  (ill-conditioned landmark Gram) they perturb the gradient at the 1e-3 relative level (see test comments).
* See `docs/reproduction_report.md` for unverified configurations.
