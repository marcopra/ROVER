"""Closed-form occupancy objective and policy gradient for standard (dense) ROVER.

Notation follows ``rover.pdf``.  All quantities are *empirical*: the transition
operator is the sink-augmented kernel ridge estimator of Eq. (17), written in the
"sandwich" form T = Phi^T G Psi of Eq. (13) / Prop. 4 with G = (K + lambda I)^-1.

Given ``n`` transitions (x_i, a_i, x'_i):

* ``K``      (n, n)     state-action Gram, K_ij = k(x_i, x_j) 1[a_i = a_j]  (Thm. 2)
* ``M``      (n, n)     policy matrix,     M_ij = k(x_i, x'_j) pi(a_i | x'_j)
* ``alpha``  (n,)       initial embedding mu_0 = Phi^T alpha (one-hot on the first sample)
* ``gram``   (n+1, n+1) Gram of the sink-augmented next-state features (``sink_residual_gram``)

With the sink state appended, the resolvent coefficients solve

    (I - gamma * blockdiag(G M, 1)) z = [alpha; 1]                    (Prop. 4, Thm. 5)

the occupancy embedding is mu = (1 - gamma) Phi~^T z, hence the objective is

    J(pi) = ||mu||^2 = (1 - gamma)^2 z^T gram z                       (Eq. 8)

and the dual gradient coefficients of Thm. 2 are

    c = 2 gamma (1 - gamma)^2 blockdiag(G, 1) (I - gamma blockdiag(G M, 1))^-T gram z.

Efficiency notes: ``K + lambda I`` is Cholesky-factorised once per actor update;
every resolvent matrix is LU-factorised once and shared by the loss and the
gradient (via the adjoint solve), so no matrix is factorised twice and no
explicit feature matrix of the augmented space is ever formed.
"""

from __future__ import annotations

import torch


def sink_residual_gram(
    state_gram: torch.Tensor, sink_coefficients: torch.Tensor, sink_norm: float
) -> torch.Tensor:
    """Gram matrix of the sink-augmented next-state features (Eq. 16-17).

    The shifted ridge target of Eq. (17) replaces phi(x'_i) by the residual
    r_i = (phi(x'_i), -eps * c_i) in H + R, with c_i = <1, psi(x_i, a_i)> (the
    unit mass of the source features) and the sink feature e = (0, eps).  With
    ``state_gram`` = Phi' Phi'^T this gives

        <r_i, r_j> = state_gram_ij + eps^2 c_i c_j,
        <r_i, e>   = -eps^2 c_i,        <e, e> = eps^2.

    A nonlinear kernel applied to concatenated residual coordinates would *not*
    preserve this algebra, so the Gram is always assembled from ``state_gram``.
    """
    n = state_gram.shape[0]
    assert state_gram.shape == (n, n) and sink_coefficients.shape == (n,)
    eps_sq = float(sink_norm) ** 2
    c = sink_coefficients.reshape(n, 1)
    gram = state_gram.new_empty(n + 1, n + 1)
    gram[:n, :n] = state_gram
    gram[:n, :n].addmm_(c, c.T, alpha=eps_sq)
    gram[:n, n:] = -eps_sq * c
    gram[n:, :n] = -eps_sq * c.T
    gram[n, n] = eps_sq
    return gram


class ResolventSolution:
    """Occupancy coefficients ``z`` of one policy, with a reusable LU factorisation."""

    def __init__(self, model: "DenseOccupancyModel", lu: torch.Tensor, pivots: torch.Tensor):
        self._model = model
        self._lu = lu
        self._pivots = pivots
        rhs = model.alpha_augmented
        self.z = torch.linalg.lu_solve(lu, pivots, rhs)
        gamma = model.gamma
        self._gram_z = model.gram @ self.z
        self.loss = (1.0 - gamma) ** 2 * (self.z * self._gram_z).sum()

    def gradient_coefficients(self) -> torch.Tensor:
        """Dual coefficients ``c`` (n+1, 1) of Thm. 2 for this policy."""
        model = self._model
        n = model.n
        gamma = model.gamma
        w = torch.linalg.lu_solve(self._lu, self._pivots, self._gram_z, adjoint=True)
        c = torch.empty_like(w)
        c[:n] = torch.cholesky_solve(w[:n], model.cholesky)
        c[n:] = w[n:]
        return (2.0 * gamma * (1.0 - gamma) ** 2) * c


class DenseOccupancyModel:
    """Sink-augmented ROVER occupancy model fixed for one actor update."""

    def __init__(
        self,
        state_action_gram: torch.Tensor,
        coverage_gram: torch.Tensor,
        alpha: torch.Tensor,
        gamma: float,
        lambda_reg: float,
    ):
        n = state_action_gram.shape[0]
        assert 0.0 < gamma < 1.0 and lambda_reg > 0.0
        assert state_action_gram.shape == (n, n)
        assert coverage_gram.shape == (n + 1, n + 1) and alpha.shape == (n,)
        self.n = n
        self.gamma = float(gamma)
        self.gram = coverage_gram
        regularised = state_action_gram.clone()
        regularised.diagonal().add_(lambda_reg)
        # Raises torch.linalg.LinAlgError if K + lambda I is not positive definite.
        self.cholesky = torch.linalg.cholesky(regularised)
        self.alpha_augmented = torch.cat([alpha, alpha.new_ones(1)]).reshape(n + 1, 1)

    def solve(self, policy_matrix: torch.Tensor) -> ResolventSolution:
        """Solve the resolvent system for ``M`` (n, n)."""
        n = self.n
        assert policy_matrix.shape == (n, n)
        identity = torch.eye(n, dtype=policy_matrix.dtype, device=policy_matrix.device)
        # Out-of-place so that autograd through the objective stays valid.
        resolvent = identity - self.gamma * torch.cholesky_solve(policy_matrix, self.cholesky)
        # Sink block of I - gamma * blockdiag(GM, 1) is 1 - gamma (absorbing state).
        augmented = torch.nn.functional.pad(resolvent, (0, 1, 0, 1))
        augmented[n, n] = 1.0 - self.gamma
        lu, pivots = torch.linalg.lu_factor(augmented)
        return ResolventSolution(self, lu, pivots)
