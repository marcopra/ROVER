"""Correctness tests for the dense ROVER occupancy objective and gradient."""

import pytest
import torch
import torch.nn.functional as F

from agent.rover_math import DenseOccupancyModel, sink_residual_gram

DTYPE = torch.float64
GAMMA, LAMBDA, SINK_NORM = 0.9, 1e-3, 0.3


def make_problem(n=14, d=5, n_actions=3, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    phi = F.normalize(torch.rand(n, d, generator=g, dtype=DTYPE), p=1, dim=1)
    phi_next = F.normalize(torch.rand(n, d, generator=g, dtype=DTYPE), p=1, dim=1)
    actions = torch.randint(0, n_actions, (n,), generator=g)
    one_hot = F.one_hot(actions, n_actions).to(DTYPE)
    alpha = torch.zeros(n, dtype=DTYPE)
    alpha[0] = 1.0
    H = phi @ phi_next.T                                      # H_ij = k(x_i, x'_j)
    K = (phi @ phi.T) * (actions[:, None] == actions[None, :])
    sink_coefficients = phi.sum(1)                            # <1, psi(x_i, a_i)> = 1
    logits = torch.randn(n, n_actions, generator=g, dtype=DTYPE)
    to_device = lambda t: t.to(device)
    return dict(
        phi=to_device(phi), phi_next=to_device(phi_next), actions=actions, E=to_device(one_hot),
        alpha=to_device(alpha), H=to_device(H), K=to_device(K),
        sink_coefficients=to_device(sink_coefficients), logits=to_device(logits),
    )


def policy_matrix(p, pi):
    """M_ij = k(x_i, x'_j) pi(a_i | x'_j)."""
    return p["H"] * (p["E"] @ pi.T)


def build_model(p):
    gram = sink_residual_gram(p["phi_next"] @ p["phi_next"].T, p["sink_coefficients"], SINK_NORM)
    return DenseOccupancyModel(p["K"], gram, p["alpha"], GAMMA, LAMBDA)


def legacy_reference(p, M):
    """Explicit-feature implementation used by the original release (agent/rover.py v1)."""
    n, d = p["phi"].shape
    eye = torch.eye
    psi = torch.einsum("bd,ba->bda", p["phi"], p["E"]).reshape(n, -1)
    phi_next = torch.cat([p["phi_next"], torch.zeros(n, 1, dtype=DTYPE, device=M.device)], 1)
    sink = torch.zeros(d + 1, 1, dtype=DTYPE, device=M.device)
    sink[-1] = SINK_NORM
    upper = phi_next.T - sink @ torch.ones(1, psi.shape[1], dtype=DTYPE, device=M.device) @ psi.T
    tilde_phi_T = torch.zeros(d + 2, n + 1, dtype=DTYPE, device=M.device)
    tilde_phi_T[: d + 1, :n] = upper
    tilde_phi_T[: d + 1, -1:] = sink
    tilde_phi = tilde_phi_T.T
    A = p["K"] + LAMBDA * eye(n, dtype=DTYPE, device=M.device)
    BM = torch.linalg.solve(A, M)
    tBM = torch.zeros(n + 1, n + 1, dtype=DTYPE, device=M.device)
    tBM[:n, :n] = BM
    tBM[n, n] = 1.0
    tA = torch.zeros(n + 1, n + 1, dtype=DTYPE, device=M.device)
    tA[:n, :n] = A
    tA[n, n] = 1.0
    alpha = torch.cat([p["alpha"], p["alpha"].new_ones(1)]).reshape(-1, 1)
    R = eye(n + 1, dtype=DTYPE, device=M.device) - GAMMA * tBM
    z = torch.linalg.solve(R, alpha)
    nu = (1 - GAMMA) * tilde_phi.T @ z
    loss = (nu ** 2).sum()
    sym = torch.linalg.solve(R.T, tilde_phi)
    left = torch.linalg.solve(tA.T, sym)
    gradient = 2 * GAMMA * (1 - GAMMA) ** 2 * left @ (sym.T @ alpha)
    return loss, gradient


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_and_gradient_match_explicit_feature_reference(seed):
    p = make_problem(seed=seed)
    pi = torch.softmax(p["logits"], dim=1)
    M = policy_matrix(p, pi)
    solution = build_model(p).solve(M)
    loss_ref, grad_ref = legacy_reference(p, M)
    torch.testing.assert_close(solution.loss, loss_ref, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(solution.gradient_coefficients(), grad_ref, rtol=1e-9, atol=1e-11)


def test_gradient_matches_autograd_of_the_objective():
    """Thm. 2 gradient vs. autograd of J through the policy table.

    Autograd differentiates J with respect to the table pi(a | x'_j).  The
    functional (RKHS) gradient of Thm. 2 is that table gradient divided by the
    occupancy coefficient z_j of the state x'_j (the occupancy weighting of the
    policy-gradient theorem), up to an action-independent constant per row.
    """
    p = make_problem(n=12, seed=3)
    model = build_model(p)
    pi = torch.softmax(p["logits"], dim=1).requires_grad_(True)
    solution = model.solve(policy_matrix(p, pi))
    solution.loss.backward()
    table_gradient = pi.grad

    with torch.no_grad():
        c = model.solve(policy_matrix(p, pi.detach())).gradient_coefficients()
        z = model.solve(policy_matrix(p, pi.detach())).z[:-1, 0]
        functional = p["H"].T @ (c[:-1] * p["E"])            # sum_i c_i k(x_i, x'_j) 1[a_i = a]
    center = lambda t: t - t.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        center(table_gradient), center(z[:, None] * functional), rtol=1e-8, atol=1e-10
    )


def test_objective_decomposes_into_support_and_sink_mass():
    """Thm. 5: ||mu||^2 = ||mu_hat||^2 + eps^2 m_e^2 (sink mass is orthogonal)."""
    p = make_problem(seed=4)
    pi = torch.softmax(p["logits"], dim=1)
    model = build_model(p)
    z = model.solve(policy_matrix(p, pi)).z
    state_gram = p["phi_next"] @ p["phi_next"].T
    # Coefficients on the real next-state features and on the sink feature.
    mu_hat_sq = (1 - GAMMA) ** 2 * (z[:-1].T @ state_gram @ z[:-1])
    sink_mass = (1 - GAMMA) * (z[-1] - p["sink_coefficients"] @ z[:-1, 0])
    total = model.solve(policy_matrix(p, pi)).loss
    torch.testing.assert_close(total, mu_hat_sq.squeeze() + SINK_NORM ** 2 * sink_mass.squeeze() ** 2,
                               rtol=1e-10, atol=1e-12)


def test_sink_gram_is_positive_semidefinite_and_symmetric():
    p = make_problem(seed=5)
    gram = sink_residual_gram(p["phi_next"] @ p["phi_next"].T, p["sink_coefficients"], SINK_NORM)
    torch.testing.assert_close(gram, gram.T)
    assert torch.linalg.eigvalsh(gram).min() > -1e-12


def test_float32_gradient_is_close_to_float64():
    """Numerical stability: the float32 path used in training stays close to float64."""
    p = make_problem(n=40, d=8, seed=6)
    pi = torch.softmax(p["logits"], dim=1)
    ref = build_model(p).solve(policy_matrix(p, pi))
    p32 = {k: (v.float() if v.is_floating_point() else v) for k, v in p.items()}
    gram32 = sink_residual_gram(p32["phi_next"] @ p32["phi_next"].T, p32["sink_coefficients"], SINK_NORM)
    model32 = DenseOccupancyModel(p32["K"], gram32, p32["alpha"], GAMMA, LAMBDA)
    sol32 = model32.solve(policy_matrix(p32, pi.float()))
    torch.testing.assert_close(sol32.loss.double(), ref.loss, rtol=1e-3, atol=1e-6)
    g32, g64 = sol32.gradient_coefficients().double(), ref.gradient_coefficients()
    assert (g32 - g64).norm() / g64.norm() < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_consistency():
    p = make_problem(n=64, d=10, seed=7)
    pi = torch.softmax(p["logits"], dim=1)
    cpu = build_model(p).solve(policy_matrix(p, pi))
    pg = make_problem(n=64, d=10, seed=7, device="cuda")
    gpu = build_model(pg).solve(policy_matrix(pg, pi.cuda()))
    torch.testing.assert_close(gpu.loss.cpu(), cpu.loss, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(gpu.gradient_coefficients().cpu(), cpu.gradient_coefficients(),
                               rtol=1e-7, atol=1e-9)


def test_non_positive_definite_gram_fails_loudly():
    p = make_problem(seed=8)
    bad = p["K"].clone()
    bad[0, 0] = -10.0
    with pytest.raises(torch.linalg.LinAlgError):
        gram = sink_residual_gram(p["phi_next"] @ p["phi_next"].T, p["sink_coefficients"], 0.1)
        DenseOccupancyModel(bad, gram, p["alpha"], GAMMA, LAMBDA)
