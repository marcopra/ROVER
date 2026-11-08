"""Nystrom approximation of the sink-augmented ROVER occupancy objective and its gradient.

The dense model of ``agent/rover_math.py`` needs ``n x n`` factorisations.  Here the
transition operator is estimated on ``m << n`` landmark state-actions instead
(paper Sec. 3.1, "Nystrom approximation"), separately for each action ``a``:

    A_a = lambda K_aa + sum_{i: a_i = a} k(x_i, S_a)^T k(x_i, S_a)         (m_a x m_a)
    T_a = A_a^{-1}      sum_{i: a_i = a} k(x_i, S_a)^T k(x_i, X')          (m_a x m)

where ``S_a`` are the landmarks with action ``a``, ``X'`` the landmarks' next states, and
``k(x_i, .)`` the dynamics kernel on the learned features.  For a policy ``pi`` the landmark-basis
transition matrix is ``T(pi)[j, l] = T_{a_j}[j, l] pi(a_j | x'_l)`` and the occupancy
coefficients solve ``(I - gamma blockdiag(T(pi), 1)) z = [alpha; 1]`` exactly as in the dense case
(``z`` lives on the landmarks' next states plus the absorbing sink).  The objective is then

    J(pi) = (1 - gamma)^2 z^T G z,   G = sink-augmented *coverage* Gram of the landmark next states

which is where the subspace enters: ``G`` is built with the coverage kernel on the projection
``g(s) = (x, y)`` while the dynamics kernel above sees the full state (paper App. E).

The gradient coefficients (one per actor-batch row plus the sink) use a reduced-rank adjoint
solve: the resolvent is restricted to the span of the top-``rank`` eigenvectors of the landmark
Gram, which bounds the cost of the transposed solve by ``rank^3`` instead of ``m^3``.  With
``rank >= m`` the reduction is exact (up to the fixed ridge jitters below).
"""

import torch

from agent.kernels import Kernel

_EIGEN_JITTER = 1e-8      # stabilises the landmark eigendecomposition
_NORMAL_JITTER = 1e-6     # stabilises the per-action normal equations A_a
_ADJOINT_JITTER = 1e-6    # stabilises the reduced adjoint system and the sink coefficient


def resolvent_matrix(transition: torch.Tensor, gamma: float) -> torch.Tensor:
    """I - gamma blockdiag(T, 1), with sink diagonal 1 - gamma (absorbing sink state)."""
    m = transition.shape[0]
    out = torch.eye(m + 1, dtype=transition.dtype, device=transition.device)
    out[:m, :m] -= gamma * transition
    out[m, m] = 1.0 - gamma
    return out


def solve_occupancy_coefficients(transition: torch.Tensor, alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    """z with (I - gamma blockdiag(T, 1)) z = [alpha; 1], shape (m + 1, 1)."""
    rhs = torch.cat([alpha.reshape(-1, 1), alpha.new_ones(1, 1)])
    return torch.linalg.solve(resolvent_matrix(transition, gamma), rhs)


class NystromOccupancyModel:
    """Block-wise Nystrom regression with cached factorisations for one actor update."""

    @torch.no_grad()
    def __init__(
        self,
        kernel: Kernel,
        states: torch.Tensor,
        actions: torch.Tensor,
        landmarks: torch.Tensor,
        landmark_actions: torch.Tensor,
        next_landmarks: torch.Tensor,
        alpha: torch.Tensor,
        coverage_gram: torch.Tensor,
        gamma: float,
        lambda_reg: float,
        rank: int | None,
        block_size: int = 1024,
    ):
        n, m = states.shape[0], landmarks.shape[0]
        n_actions = int(max(actions.max(), landmark_actions.max())) + 1
        assert n > 0 and m > 0 and 0.0 < gamma < 1.0 and lambda_reg > 0.0 and block_size > 0
        assert landmarks.shape == next_landmarks.shape == (m, states.shape[1])
        assert alpha.shape == (m, 1) and coverage_gram.shape == (m + 1, m + 1)
        assert actions.shape == (n,) and landmark_actions.shape == (m,)
        assert rank is None or rank > 0
        for tensor in (states, landmarks, next_landmarks, alpha, coverage_gram):
            assert tensor.dtype == states.dtype and tensor.device == states.device
            assert torch.isfinite(tensor).all()
        self.kernel, self.gamma, self.n_actions = kernel, float(gamma), n_actions
        self.states, self.actions = states, actions
        self.landmarks, self.next_landmarks = landmarks, next_landmarks
        self.alpha, self.gram, self.block_size = alpha, coverage_gram, block_size

        rank = m if rank is None else min(m, int(rank))
        self.blocks, spectra = [], []
        for action in landmark_actions.unique(sorted=True).tolist():
            rows = torch.nonzero(actions == action, as_tuple=True)[0]
            columns = torch.nonzero(landmark_actions == action, as_tuple=True)[0]
            support = landmarks[columns]
            gram_ss = self._kernel_rows(support, support)
            values, vectors = torch.linalg.eigh(gram_ss + _EIGEN_JITTER * torch.eye(
                len(columns), dtype=gram_ss.dtype, device=gram_ss.device))
            keep = min(rank, len(columns))
            spectra.append((values[-keep:].clone(), vectors[:, -keep:].clone()))
            normal = lambda_reg * gram_ss
            cross = states.new_zeros(len(columns), m)
            for start in range(0, len(rows), block_size):
                batch = states[rows[start:start + block_size]]
                k_bs = kernel(batch, support)
                normal.addmm_(k_bs.T, k_bs)
                cross.addmm_(k_bs.T, kernel(batch, next_landmarks))
            normal.diagonal().add_(_NORMAL_JITTER)
            lu, pivots = torch.linalg.lu_factor(normal)
            self.blocks.append(dict(
                action=action, rows=rows, columns=columns, lu=lu, pivots=pivots,
                transition=torch.linalg.lu_solve(lu, pivots, cross),
            ))

        # Global top-`rank` eigenspace of the block-diagonal landmark Gram.
        all_values = torch.cat([values for values, _ in spectra])
        selected = torch.argsort(all_values, stable=True)[-rank:]
        self.basis_matrix = states.new_zeros(m, rank)   # U: (m, rank)
        offset = 0
        for block, (values, vectors) in zip(self.blocks, spectra):
            local = torch.nonzero((selected >= offset) & (selected < offset + len(values)), as_tuple=True)[0]
            local_u = vectors[:, selected[local] - offset]
            self.basis_matrix[block["columns"][:, None], local] = local_u
            block["basis"] = local
            block["adjoint_projection"] = torch.linalg.lu_solve(block["lu"], block["pivots"], local_u, adjoint=True)
            block["projected_transition"] = local_u.T @ block["transition"]
            offset += len(values)
        U = self.basis_matrix
        self.basis_gram = U.T @ U   # ~ I; kept exact because eigenvectors are only approximately orthonormal
        self.coverage_rr = U.T @ coverage_gram[:-1, :-1] @ U
        self.coverage_re = U.T @ coverage_gram[:-1, -1:]
        self.coverage_er = coverage_gram[-1:, :-1] @ U
        self.alpha_r = U.T @ alpha

    def _kernel_rows(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        out = X.new_empty(len(X), len(Y))
        for start in range(0, len(X), self.block_size):
            out[start:start + self.block_size] = self.kernel(X[start:start + self.block_size], Y)
        return out

    def _solve_normal(self, rhs: torch.Tensor, adjoint: bool = False) -> torch.Tensor:
        result = torch.empty_like(rhs)
        for block in self.blocks:
            columns = block["columns"]
            result[columns] = torch.linalg.lu_solve(block["lu"], block["pivots"], rhs[columns], adjoint=adjoint)
        return result

    def _apply_kernel(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Row i -> k(x_i, S_{a_i}) @ coefficients[S_{a_i}] (state-action kernel, action-masked)."""
        result = self.states.new_zeros(len(self.states), coefficients.shape[1])
        for block in self.blocks:
            rows, columns = block["rows"], block["columns"]
            support, coeff = self.landmarks[columns], coefficients[columns]
            for start in range(0, len(rows), self.block_size):
                batch = rows[start:start + self.block_size]
                result[batch] = self.kernel(self.states[batch], support) @ coeff
        return result

    @torch.no_grad()
    def policy(self, coefficients: torch.Tensor) -> torch.Tensor:
        """PMD policy pi(. | x'_l) at the landmarks' next states: softmax(-logits) (Eq. 15)."""
        assert coefficients.shape == (len(self.states) + 1, 1)
        one_hot = torch.nn.functional.one_hot(self.actions, self.n_actions).to(self.states.dtype)
        logits = self.states.new_zeros(len(self.landmarks), self.n_actions)
        for start in range(0, len(self.states), self.block_size):
            rows = slice(start, start + self.block_size)
            h = self.kernel(self.states[rows], self.next_landmarks)
            logits.addmm_(h.T, coefficients[:-1][rows] * one_hot[rows])
        logits.add_(coefficients[-1])
        assert torch.isfinite(logits).all(), "non-finite Nystrom policy logits"
        return torch.softmax(-logits, dim=1)

    @torch.no_grad()
    def transition_matrix(self, pi: torch.Tensor) -> torch.Tensor:
        """T(pi) in the landmark basis, shape (m, m)."""
        assert pi.shape == (len(self.landmarks), self.n_actions)
        assert torch.isfinite(pi).all() and (pi >= 0).all() and torch.allclose(pi.sum(1), torch.ones_like(pi[:, 0]))
        transition = self.states.new_empty(len(self.landmarks), len(self.landmarks))
        for block in self.blocks:
            transition[block["columns"]] = block["transition"] * pi[:, block["action"]]
        return transition

    @torch.no_grad()
    def loss(self, pi: torch.Tensor) -> torch.Tensor:
        """J(pi) = (1 - gamma)^2 z^T G z."""
        z = solve_occupancy_coefficients(self.transition_matrix(pi), self.alpha, self.gamma)
        return ((1.0 - self.gamma) ** 2 * (z.T @ self.gram @ z)).squeeze()

    @torch.no_grad()
    def gradient_coefficients(self, pi: torch.Tensor) -> torch.Tensor:
        """Dual coefficients c (n + 1, 1) of Thm. 2 via the reduced-rank adjoint solve."""
        gamma = self.gamma
        assert pi.shape == (len(self.landmarks), self.n_actions)
        system = self.basis_gram.clone()                      # S_r = U^T U - gamma U^T T(pi) U
        for block in self.blocks:
            weighted_u = pi[:, block["action"]:block["action"] + 1] * self.basis_matrix
            system[block["basis"]] -= gamma * (block["projected_transition"] @ weighted_u)
        z_r = torch.linalg.solve(system, self.alpha_r)
        system.diagonal().add_(_ADJOINT_JITTER)
        z_sink = 1.0 / (1.0 - gamma)                          # absorbing sink: (1 - gamma) z_e = 1
        h_r = self.coverage_rr @ z_r + self.coverage_re * z_sink
        h_e = self.coverage_er @ z_r + self.gram[-1:, -1:] * z_sink
        adjoint = torch.linalg.solve(system.T, h_r)
        solved = self.states.new_empty(len(self.landmarks), 1)
        for block in self.blocks:
            solved[block["columns"]] = block["adjoint_projection"] @ adjoint[block["basis"]]
        gradient = self.states.new_empty(len(self.states) + 1, 1)
        gradient[:-1] = self._apply_kernel(solved)
        gradient[-1:] = h_e / (1.0 - gamma + _ADJOINT_JITTER)
        gradient.mul_(2.0 * gamma * (1.0 - gamma) ** 2)
        assert torch.isfinite(gradient).all(), "non-finite Nystrom gradient"
        return gradient
