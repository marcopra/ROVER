"""Subspace ROVER: occupancy coverage on a projection of the state, with Nystrom acceleration.

Relation to standard ROVER (paper App. E).  Standard ROVER minimises ``||mu_{d^pi}||^2`` where the
same learned features ``phi`` define both the transition operator and the coverage kernel.
Subspace ROVER separates the two:

* the **dynamics kernel** ``k_d`` (on learned features of the full state, e.g. (x, y, vx, vy))
  parameterises the transition operator / resolvent, so the world model sees the full state;
* the **coverage kernel** ``k_c`` acts on the projection ``g(s)`` (e.g. position (x, y) only) and
  defines the objective ``J_c(pi) = ||mu_c^pi||^2 = ∬ k_c(g(s), g(s~)) dd^pi dd^pi``.

Two states with the same position but different velocity therefore count as the same location.
When ``g`` is the identity and ``k_c = k_d`` this is exactly standard ROVER; the code path only
differs in the Gram matrix ``G`` that enters ``J = (1 - gamma)^2 z^T G z`` (``NystromOccupancyModel``).

Because a continuous state space needs many samples, the operator is estimated on ``subsamples``
Nystrom landmarks chosen by pivoted Cholesky (``agent/landmarks.py``), which replaces the
``O(n^3)`` solves of the dense model by ``O(m^3)`` ones (paper Sec. 3.1).

This module implements the state-observation variant used for the PointMaze results.
"""

import copy
import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.kernels import Kernel, median_bandwidth
from agent.landmarks import pivoted_cholesky
from agent.nystrom import NystromOccupancyModel
from agent.rover_math import sink_residual_gram
from agent.rover_networks import Encoder, ProjectSA

logger = logging.getLogger(__name__)


class SubspaceRoverAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        coverage_indices,
        discount: float,
        lambda_reg: float,
        lr_actor: float,
        pmd_steps: int,
        sink_schedule: str,
        epsilon_schedule,
        feature_dim: int = 256,
        hidden_dim: int = 1024,
        lr_encoder: float = 1e-4,
        lr_T: float = 1e-4,
        batch_size: int = 1024,
        batch_size_actor: int = 100_000,
        subsamples: int = 10_000,
        pca_truncation: Optional[int] = 1000,
        candidate_multiplier: float = 5.0,
        cholesky_tolerance: float = 1e-6,
        dynamics_kernel: str = "gaussian",
        dynamics_bandwidth_mult: float = 1.0,
        coverage_bandwidth_mult: float = 1.0,
        update_every_steps: int = 10,
        update_actor_every_steps: int = 20_000,
        T_init_steps: int = 5_000,
        num_expl_steps: int = 10_000,
        total_train_steps: int = 250_000,
        pmd_eta_min: float = 1e-8,
        pmd_eta_max: float = 1e8,
        pmd_backtrack_factor: float = 0.25,
        pmd_backtrack_max_trials: int = 12,
        capacity: int = 1_000_000,
        device: str = "cuda",
        compute_dtype: torch.dtype = torch.float64,
    ):
        assert subsamples <= batch_size_actor, "cannot select more landmarks than actor samples"
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.coverage_indices = torch.as_tensor(list(coverage_indices), dtype=torch.long, device=device)
        self.discount, self.lambda_reg = discount, lambda_reg
        self.lr_actor, self.pmd_steps = lr_actor, pmd_steps
        self.sink_schedule, self.epsilon_schedule = sink_schedule, epsilon_schedule
        self.batch_size, self.batch_size_actor = batch_size, batch_size_actor
        self.subsamples, self.pca_truncation = subsamples, pca_truncation
        self.candidate_multiplier, self.cholesky_tolerance = candidate_multiplier, cholesky_tolerance
        self.dynamics_bandwidth_mult = dynamics_bandwidth_mult
        self.coverage_bandwidth_mult = coverage_bandwidth_mult
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.T_init_steps, self.num_expl_steps = T_init_steps, num_expl_steps
        self.pmd_eta_min, self.pmd_eta_max = pmd_eta_min, pmd_eta_max
        self.pmd_backtrack_factor = pmd_backtrack_factor
        self.pmd_backtrack_max_trials = pmd_backtrack_max_trials
        self.device, self.dtype = device, compute_dtype

        # Learned dynamics representation: unit-L1 features of the full state, trained with InfoNCE.
        self.encoder = Encoder((obs_dim,), hidden_dim, feature_dim).to(device)
        self.project_sa = ProjectSA(feature_dim * n_actions, hidden_dim, feature_dim, linear=True).to(device)
        self.policy_encoder = copy.deepcopy(self.encoder).eval().requires_grad_(False)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr_encoder, weight_decay=1e-5)
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer, T_max=max(total_train_steps, 1), eta_min=0.1 * lr_encoder)
        self.transition_optimizer = torch.optim.Adam(self.project_sa.parameters(), lr=lr_T)

        self.dynamics_kernel = Kernel(dynamics_kernel, None)
        self.coverage_kernel = Kernel("gaussian", None)

        # Raw transition store (observations are 4-dimensional; keep everything on the device).
        self._obs = torch.empty(capacity, obs_dim, device=device)
        self._next_obs = torch.empty(capacity, obs_dim, device=device)
        self._action = torch.empty(capacity, dtype=torch.long, device=device)
        self.size = 0

        # Acting state, refreshed by every actor update.
        self._actor_features: Optional[torch.Tensor] = None   # (n, d) features of the actor batch
        self._actor_weights: Optional[torch.Tensor] = None    # (n, |A|) c_i * 1[a_i = a]
        self._sink_logit = 0.0

    # ------------------------------------------------------------------ data
    def observe(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> None:
        assert self.size < len(self._obs), "transition store is full"
        self._obs[self.size] = torch.as_tensor(obs, device=self.device)
        self._next_obs[self.size] = torch.as_tensor(next_obs, device=self.device)
        self._action[self.size] = int(action)
        self.size += 1

    def train(self, training: bool = True) -> None:
        self.encoder.train(training)
        self.project_sa.train(training)
        self.policy_encoder.eval()

    # ---------------------------------------------------------------- acting
    @torch.no_grad()
    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """pi(. | x) for observations of shape (B, obs_dim) under the current PMD iterate."""
        obs = np.atleast_2d(obs)
        if self._actor_features is None:
            return np.full((len(obs), self.n_actions), 1.0 / self.n_actions)
        phi = self.policy_encoder(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).to(self.dtype)
        logits = self.dynamics_kernel(phi, self._actor_features) @ self._actor_weights + self._sink_logit
        probs = torch.softmax(-logits, dim=1)
        assert torch.isfinite(probs).all(), "non-finite action probabilities"
        return probs.cpu().numpy()

    def act(self, obs: np.ndarray, step: int) -> int:
        """Epsilon-greedy over the PMD policy; uniform random during the seed phase."""
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)
        probs = self.action_probabilities(obs)[0].astype(np.float64)
        return int(np.random.choice(self.n_actions, p=probs / probs.sum()))

    # -------------------------------------------------------------- learning
    def _initialised(self, step: int) -> bool:
        return step >= self.num_expl_steps + self.T_init_steps

    def update(self, step: int) -> Dict[str, float]:
        metrics = {}
        if self.size < self.batch_size:
            return metrics
        initialised = self._initialised(step)
        if (not initialised) or step % self.update_every_steps == 0:
            metrics.update(self.update_encoders())
        if initialised and (step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps):
            metrics.update(self.update_actor(step))
        return metrics

    def update_encoders(self) -> Dict[str, float]:
        """InfoNCE on (x, a) -> x' with in-batch negatives (paper Eq. 9)."""
        index = torch.randint(0, self.size, (self.batch_size,), device=self.device)
        obs, action, next_obs = self._obs[index], self._action[index], self._next_obs[index]
        features = self.encoder(obs)
        with torch.no_grad():
            target = self.encoder(next_obs)
        one_hot = F.one_hot(action, self.n_actions).to(features.dtype)
        sa = torch.einsum("bd,ba->bda", features, one_hot).reshape(self.batch_size, -1)
        logits = F.normalize(self.project_sa(sa), dim=1, eps=1e-10) @ F.normalize(target, dim=1, eps=1e-10).T
        logits = logits - logits.max(dim=1, keepdim=True).values
        loss = F.cross_entropy(logits, torch.arange(self.batch_size, device=self.device))
        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.transition_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_optimizer.step()
        self.transition_optimizer.step()
        self.encoder_scheduler.step()
        return {"transition_loss": loss.item()}

    @torch.no_grad()
    def _encode(self, states: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        return torch.cat([self.policy_encoder(states[i:i + chunk]) for i in range(0, len(states), chunk)]).to(self.dtype)

    @torch.no_grad()
    def update_actor(self, step: int) -> Dict[str, float]:
        """Fit the Nystrom world model on the latest transitions and run PMD (Alg. 1, lines 5-12)."""
        self.policy_encoder.load_state_dict(self.encoder.state_dict())
        self.policy_encoder.eval().requires_grad_(False)

        # Actor batch: a pinned reset self-transition (initial mass at the reset state) followed by
        # the most recent transitions, so the batch always reflects the current visitation.
        recent = torch.arange(max(1, self.size - (self.batch_size_actor - 1)), self.size, device=self.device)
        obs = torch.cat([self._obs[:1], self._obs[recent]])
        next_obs = torch.cat([self._obs[:1], self._next_obs[recent]])
        actions = torch.cat([self._action[:1], self._action[recent]])
        n = len(obs)
        phi, phi_next = self._encode(obs), self._encode(next_obs)

        # Landmarks: pivoted Cholesky over a random candidate subset (the pinned row is forced first).
        count = min(self.subsamples, n)
        n_candidates = min(n, max(count, int(np.ceil(self.candidate_multiplier * count))))
        candidates = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device),
                                torch.randperm(n - 1, device=self.device)[:n_candidates - 1] + 1])
        if self.dynamics_kernel.kind == "gaussian":
            self.dynamics_kernel.bandwidth = median_bandwidth(phi[candidates], self.dynamics_bandwidth_mult)
        chosen = pivoted_cholesky(phi[candidates], actions[candidates], count, self.dynamics_kernel,
                                  tolerance=self.cholesky_tolerance, force_first=True).to(self.device)
        landmarks = candidates[chosen]

        # Coverage Gram on the projection g(s) of the landmarks' next states, plus the sink (Eq. 16-17).
        coverage_points = next_obs[landmarks][:, self.coverage_indices].to(self.dtype)
        self.coverage_kernel.bandwidth = median_bandwidth(coverage_points, self.coverage_bandwidth_mult)
        m = len(landmarks)
        sink_norm = float(utils.schedule(self.sink_schedule, step))
        gram = sink_residual_gram(self.coverage_kernel(coverage_points, coverage_points),
                                  phi[landmarks].sum(1), sink_norm)
        alpha = torch.zeros(m, 1, device=self.device, dtype=self.dtype)
        alpha[0] = 1.0   # initial embedding mu_0: the pinned reset transition
        model = NystromOccupancyModel(
            self.dynamics_kernel, phi, actions, phi[landmarks], actions[landmarks], phi_next[landmarks], alpha,
            gram, self.discount, self.lambda_reg, self.pca_truncation)

        coeff, pi, loss, rejections, eta_used = self._policy_mirror_descent(model, n, step)

        # Publish the policy: logits(x, a) = sum_i c_i k(x, x_i) 1[a_i = a] + c_sink.
        self._actor_features = phi
        self._actor_weights = coeff[:-1] * F.one_hot(actions, self.n_actions).to(self.dtype)
        self._sink_logit = float(coeff[-1])
        return {"actor_loss": float(loss), "actor_eta": eta_used, "actor_backtracking_rejections": rejections,
                "sink_norm": sink_norm, "dynamics_bandwidth": float(self.dynamics_kernel.bandwidth or 0.0),
                "coverage_bandwidth": float(self.coverage_kernel.bandwidth), "num_landmarks": m}

    def _policy_mirror_descent(self, model: NystromOccupancyModel, n: int, step: int):
        eta = float(np.clip(utils.schedule(self.lr_actor, step), self.pmd_eta_min, self.pmd_eta_max))
        coeff = torch.zeros(n + 1, 1, device=self.device, dtype=self.dtype)
        pi = model.policy(coeff)
        loss = model.loss(pi)
        assert torch.isfinite(loss), "initial occupancy objective is not finite"
        best = (loss, coeff, 0.0)
        rejections = 0
        for _ in range(self.pmd_steps):
            gradient = model.gradient_coefficients(pi)
            trial_eta = eta
            for _trial in range(self.pmd_backtrack_max_trials + 1):
                candidate = coeff + trial_eta * gradient
                candidate_pi = model.policy(candidate)
                candidate_loss = model.loss(candidate_pi)
                if torch.isfinite(candidate_loss) and candidate_loss <= loss:
                    break
                trial_eta = float(np.clip(trial_eta * self.pmd_backtrack_factor, self.pmd_eta_min, self.pmd_eta_max))
            else:
                rejections += 1   # keep the last accepted policy; the same direction cannot help
                break
            coeff, pi, loss = candidate, candidate_pi, candidate_loss
            if loss < best[0]:
                best = (loss, coeff, trial_eta)
        loss, coeff, eta_used = best   # best-iterate restoration
        return coeff, model.policy(coeff), loss, rejections, eta_used

    # ------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        return dict(
            encoder=self.encoder.state_dict(), policy_encoder=self.policy_encoder.state_dict(),
            actor_features=None if self._actor_features is None else self._actor_features.float().cpu(),
            actor_weights=None if self._actor_weights is None else self._actor_weights.float().cpu(),
            sink_logit=self._sink_logit, dynamics_bandwidth=self.dynamics_kernel.bandwidth,
        )

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.policy_encoder.load_state_dict(state["policy_encoder"])
        if state["actor_features"] is not None:
            self._actor_features = state["actor_features"].to(device=self.device, dtype=self.dtype)
            self._actor_weights = state["actor_weights"].to(device=self.device, dtype=self.dtype)
        self._sink_logit = state["sink_logit"]
        self.dynamics_kernel.bandwidth = state["dynamics_bandwidth"]
