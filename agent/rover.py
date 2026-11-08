"""Standard (dense) ROVER agent: Reward-free pretraining via Occupancy coVERage maximisation.

One iteration of Algorithm 1 / 2 of ``rover.pdf``:

1. collect transitions with the current behaviour policy ``p`` (epsilon-greedy over ``p``);
2. representation learning: InfoNCE on the encoder (Eq. 9, ``update_encoders``);
3. world model + policy mirror descent: sink-augmented kernel ridge regression
   (Eq. 17) and PMD with the dual gradient of Thm. 2 (``update_actor``).

All ``n = batch_size_actor`` transitions of the actor batch are used (no Nystrom
approximation); see ``agent/rover_subspace.py`` for the Nystrom/subspace variant.
The matrix algebra lives in ``agent/rover_math.py``.
"""

from collections import OrderedDict
import copy
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.rover_math import DenseOccupancyModel, ResolventSolution, sink_residual_gram
from agent.rover_networks import CNNEncoder, Encoder, ProjectSA  # noqa: F401  (CNNEncoder is re-exported)

logger = logging.getLogger(__name__)


class RoverAgent:
    def __init__(
        self,
        name: str,
        obs_type: str,
        obs_shape,
        action_shape,
        lr_actor: float,
        discount: float,
        lambda_reg: float,
        batch_size: int,
        batch_size_actor: int,
        nstep: int,
        use_tb: bool,
        use_wandb: bool,
        lr_T: float,
        lr_encoder: float,
        hidden_dim: int,
        feature_dim: int,
        update_every_steps: int,
        update_actor_every_steps: int,
        pmd_steps: int,
        num_expl_steps: int,
        T_init_steps: int,
        total_train_steps: int,
        sink_schedule: str,
        epsilon_schedule,
        pmd_eta_min: float = 1e-8,
        pmd_eta_max: float = 1e3,
        pmd_backtrack_factor: float = 0.25,
        pmd_backtrack_max_trials: int = 12,
        device: str = "cpu",
    ):
        assert obs_type in ("pixels", "states", "discrete_states"), obs_type
        assert batch_size_actor >= batch_size, (
            "batch_size_actor must be at least batch_size (the actor batch is "
            "assembled from encoder batches)"
        )
        self.name = name
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.n_actions = action_shape[0]
        self.feature_dim = feature_dim
        self.lr_actor = lr_actor
        self.discount = discount
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.batch_size_actor = batch_size_actor
        self.nstep = nstep  # consumed by the replay loader through the config
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.pmd_steps = pmd_steps
        self.num_expl_steps = num_expl_steps
        self.T_init_steps = T_init_steps
        self.sink_schedule = sink_schedule
        self.epsilon_schedule = epsilon_schedule
        self.pmd_eta_min = pmd_eta_min
        self.pmd_eta_max = pmd_eta_max
        self.pmd_backtrack_factor = pmd_backtrack_factor
        self.pmd_backtrack_max_trials = pmd_backtrack_max_trials
        self.device = device
        self.training = False

        if obs_type == "pixels":
            self.encoder = CNNEncoder(obs_shape, feature_dim).to(device)
        else:
            self.encoder = Encoder(obs_shape, hidden_dim, feature_dim).to(device)
        self.project_sa = ProjectSA(feature_dim * self.n_actions, hidden_dim, feature_dim).to(device)
        # Frozen snapshot of the encoder used by the actor; synchronised at every actor update
        # so that the cached features and the behaviour policy always share one representation.
        self.policy_encoder = copy.deepcopy(self.encoder)
        self._freeze(self.policy_encoder)
        self._policy_is_synced = True

        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr_encoder, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
        )
        # Cosine decay to 10% of the initial learning rate over the whole run.
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer, T_max=total_train_steps, eta_min=lr_encoder * 0.1
        )
        self.transition_optimizer = torch.optim.Adam(self.project_sa.parameters(), lr=lr_T)

        # PMD state shared with downstream transfer (agent/*_kernel_actor.py reads these attributes).
        self.gradient_coeff: Optional[torch.Tensor] = None   # (n + 1, 1): dual coefficients, last = sink
        self.E: Optional[torch.Tensor] = None                # (n, |A|) one-hot actions of the actor batch
        self._phi_all_obs: Optional[torch.Tensor] = None     # (n, d + 1) features with a zero sink column
        self.current_eta = 0.0
        self.current_action_probs = []


    # ------------------------------------------------------------------ interface
    def train(self, training: bool = True) -> None:
        self.training = training
        self.encoder.train(training)
        self.project_sa.train(training)
        self.policy_encoder.eval()

    def init_meta(self):
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def ready_for_snapshot(self) -> bool:
        return self._policy_is_synced

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _sync_policy_encoder(self) -> None:
        self.policy_encoder.load_state_dict(self.encoder.state_dict())
        self._freeze(self.policy_encoder)
        self._policy_is_synced = True

    def _encode_state_action(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """psi(x, a) = phi(x) (x) e_a, flattened to (B, d * |A|)."""
        one_hot = F.one_hot(actions.long().reshape(-1), self.n_actions).to(features.dtype)
        return torch.einsum("bd,ba->bda", features, one_hot).reshape(features.shape[0], -1)

    # --------------------------------------------------------------------- acting
    def _policy_logits(self, H: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        """PMD logits <psi(x, a), Psi^T C> for kernel rows ``H`` (Eq. 15); last coeff is the sink."""
        return H @ (coeff[:-1] * self.E) + coeff[-1]

    def compute_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """pi(. | x) for one observation under the current PMD iterate."""
        if self.gradient_coeff is None:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            phi = self.policy_encoder.encode_and_project(obs_tensor)
            phi = torch.cat([phi, phi.new_zeros(1, 1)], dim=1)  # match the zero sink column
            probs = torch.softmax(-self._policy_logits(phi @ self._phi_all_obs.T, self.gradient_coeff), dim=1)
        assert torch.isfinite(probs).all(), "non-finite action probabilities"
        return probs.cpu().numpy().flatten()

    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)
        probs = self.compute_action_probs(obs)
        self.current_action_probs.append(probs)
        return np.random.choice(self.n_actions, p=probs / probs.sum())

    # ------------------------------------------------------------------- learning
    def _is_T_sufficiently_initialized(self, step: int) -> bool:
        return step >= self.num_expl_steps + self.T_init_steps

    def update_encoders(self, obs, action, next_obs) -> Dict[str, float]:
        """InfoNCE representation loss, Eq. (9); negatives are the other next states in the batch."""
        features = self.encoder.encode_and_project(obs)
        with torch.no_grad():
            next_features = self.encoder.encode_and_project(next_obs)
        predicted = self.project_sa(self._encode_state_action(features, action))
        logits = F.normalize(predicted, p=2, dim=1, eps=1e-10) @ F.normalize(next_features, p=2, dim=1, eps=1e-10).T
        logits = logits - logits.max(dim=1, keepdim=True).values
        loss = F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))

        self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.transition_optimizer.step()
        self.encoder_scheduler.step()
        self._policy_is_synced = False
        return {"transition_loss": loss.item()} if (self.use_tb or self.use_wandb) else {}

    @torch.no_grad()
    def _cache_features(self, obs, action, next_obs):
        """Encode the actor batch with the frozen policy encoder; returns the n x n kernels.

        ``self.E`` and ``self._phi_all_obs`` are kept on the agent because downstream
        transfer (kernel actors) rebuilds the pretrained policy from them.  The n x n
        matrices are returned instead of stored so that snapshots stay small.
        """
        phi = self.policy_encoder.encode_and_project(obs)
        phi_next = self.policy_encoder.encode_and_project(next_obs)
        self.E = F.one_hot(action.long().reshape(-1), self.n_actions).to(phi.dtype)
        # Zero sink column kept for the transfer interface (inner products are unchanged).
        self._phi_all_obs = torch.cat([phi, phi.new_zeros(phi.shape[0], 1)], dim=1)
        # H_ij = k(x_i, x'_j); K_ij = k(x_i, x_j) 1[a_i = a_j] (Thm. 2).
        H = phi @ phi_next.T
        K = (phi @ phi.T) * (self.E @ self.E.T)
        # With L1-normalised features, <1, psi(x_i, a_i)> = 1 is the unit mass entering the sink residual.
        return H, K, phi_next @ phi_next.T, phi.sum(dim=1)

    def _policy_matrix(self, H: torch.Tensor, coeff: torch.Tensor):
        """Policy at the next states and M_ij = k(x_i, x'_j) pi(a_i | x'_j)."""
        pi = torch.softmax(-self._policy_logits(H.T, coeff), dim=1)
        return pi, H * (self.E @ pi.T)

    @torch.no_grad()
    def update_actor(self, obs, action, next_obs, step) -> Dict[str, float]:
        """Policy mirror descent on the sink-augmented occupancy objective (Alg. 1, lines 6-12)."""
        self._sync_policy_encoder()
        H, K, next_state_gram, sink_coefficients = self._cache_features(obs, action, next_obs)
        n = obs.shape[0]
        eta = float(np.clip(utils.schedule(self.lr_actor, step), self.pmd_eta_min, self.pmd_eta_max))
        sink_norm = float(utils.schedule(self.sink_schedule, step))

        alpha = torch.zeros(n, device=self.device, dtype=H.dtype)
        alpha[0] = 1.0  # initial embedding mu_0 = phi(x'_0): the first transition of the batch
        gram = sink_residual_gram(next_state_gram, sink_coefficients, sink_norm)
        model = DenseOccupancyModel(K, gram, alpha, self.discount, self.lambda_reg)

        coeff = torch.zeros(n + 1, 1, device=self.device, dtype=H.dtype)
        pi, M = self._policy_matrix(H, coeff)
        solution: ResolventSolution = model.solve(M)
        loss = solution.loss
        best_loss, best_coeff, best_pi, best_eta = loss, coeff, pi, 0.0
        rejections = 0
        initial_loss = float(loss)

        for iteration in range(self.pmd_steps):
            gradient = solution.gradient_coefficients()
            trial_eta = eta
            for trial in range(self.pmd_backtrack_max_trials + 1):
                candidate_coeff = coeff + trial_eta * gradient
                candidate_pi, candidate_M = self._policy_matrix(H, candidate_coeff)
                candidate = model.solve(candidate_M)
                if torch.isfinite(candidate.loss) and candidate.loss <= loss:
                    break
                trial_eta = float(np.clip(trial_eta * self.pmd_backtrack_factor, self.pmd_eta_min, self.pmd_eta_max))
            else:
                # No descent step found: keep the last accepted policy and stop; repeating the
                # same direction cannot help.
                rejections += 1
                break
            coeff, pi, solution, loss = candidate_coeff, candidate_pi, candidate, candidate.loss
            self.current_eta = trial_eta
            if loss < best_loss:
                best_loss, best_coeff, best_pi, best_eta = loss, coeff, pi, trial_eta
            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                logger.info("PMD iteration %d: occupancy norm^2 = %.6g, eta = %.3g", iteration, float(loss), trial_eta)

        # Best-iterate restoration: the deployed policy is the iterate with the lowest objective.
        self.gradient_coeff, self.pi, self.current_eta = best_coeff, best_pi, best_eta
        assert torch.isfinite(self.gradient_coeff).all(), "non-finite PMD coefficients"
        metrics = {}
        if self.use_tb or self.use_wandb:
            metrics.update(
                actor_loss=float(best_loss), actor_initial_loss=initial_loss, actor_eta=best_eta,
                actor_backtracking_rejections=rejections, sink_norm=sink_norm,
            )
        return metrics

    def update(self, replay_iter, step):
        metrics = {}
        initialised = self._is_T_sufficiently_initialized(step)
        encoder_update_due = (not initialised) or step % self.update_every_steps == 0
        actor_update_due = step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps
        if not (encoder_update_due or (initialised and actor_update_due)):
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        if encoder_update_due:
            metrics.update(self.update_encoders(obs, action, next_obs))
        if not initialised:
            return metrics
        if not actor_update_due:
            return metrics

        # Assemble the actor batch from whole encoder batches: n = (batch_size_actor // batch_size) * batch_size
        # transitions (e.g. 5000 // 1024 = 4 batches -> n = 4096, as in the published runs).
        batches = [(obs, action, next_obs)]
        for _ in range(self.batch_size_actor // self.batch_size - 1):
            obs_b, action_b, _, _, next_obs_b = utils.to_torch(next(replay_iter), self.device)
            batches.append((obs_b, action_b, next_obs_b))
        obs_actor, action_actor, next_obs_actor = (torch.cat(field, dim=0) for field in zip(*batches))
        metrics.update(self.update_actor(obs_actor, action_actor, next_obs_actor, step))
        self._log_policy_statistics(metrics)
        return metrics

    def _log_policy_statistics(self, metrics: Dict[str, float]) -> None:
        """Mean total-variation distance from uniform of the actions taken since the last update."""
        if not self.current_action_probs:
            return
        probs = np.asarray(self.current_action_probs)
        metrics["policy_tv_from_uniform"] = float(
            np.mean(0.5 * np.abs(probs - 1.0 / self.n_actions).sum(axis=-1))
        )
        self.current_action_probs = []
