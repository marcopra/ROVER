"""Tests for the Nystrom/subspace ROVER building blocks: kernels, landmarks and the occupancy model."""

import pytest
import torch
import torch.nn.functional as F

from agent.kernels import Kernel, median_bandwidth
from agent.landmarks import pivoted_cholesky
from agent.nystrom import NystromOccupancyModel
from agent.rover_math import DenseOccupancyModel, sink_residual_gram

DTYPE = torch.float64
GAMMA, LAMBDA, SINK = 0.9, 1e-2, 0.4


def make_data(n=48, d=6, n_actions=4, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    feats = F.normalize(torch.rand(n, d, generator=g, dtype=DTYPE), p=1, dim=1).to(device)
    nxt = F.normalize(torch.rand(n, d, generator=g, dtype=DTYPE), p=1, dim=1).to(device)
    xy = torch.rand(n, 2, generator=g, dtype=DTYPE).to(device)            # raw projection g(s) for coverage
    actions = torch.randint(0, n_actions, (n,), generator=g).to(device)
    return feats, nxt, xy, actions


def build(feats, nxt, xy_next, actions, rank=None, cov_bandwidth=0.3, dyn_bandwidth=0.1):
    # A modest bandwidth keeps K well conditioned: the fixed 1e-6 ridge jitters of the Nystrom solves
    # then perturb results at ~1e-5; very smooth kernels would amplify them (documented in nystrom.py).
    kernel = Kernel("gaussian", dyn_bandwidth)
    cov_kernel = Kernel("gaussian", cov_bandwidth)
    n = len(feats)
    alpha = torch.zeros(n, 1, dtype=DTYPE, device=feats.device)
    alpha[0] = 1.0
    gram = sink_residual_gram(cov_kernel(xy_next, xy_next), torch.ones(n, dtype=DTYPE, device=feats.device), SINK)
    model = NystromOccupancyModel(kernel, feats, actions, feats, actions, nxt, alpha, gram, GAMMA, LAMBDA, rank)
    return model, kernel, gram, alpha


def dense_reference(kernel, feats, nxt, actions, gram, alpha, pi):
    same = (actions[:, None] == actions[None, :]).to(DTYPE)
    K = kernel(feats, feats) * same
    H = kernel(feats, nxt)
    E = F.one_hot(actions, int(actions.max()) + 1).to(DTYPE)
    M = H * (E @ pi.T)
    return DenseOccupancyModel(K, gram, alpha.reshape(-1), GAMMA, LAMBDA).solve(M)


@pytest.mark.parametrize("seed", [0, 1])
def test_nystrom_with_all_landmarks_matches_dense_loss_and_gradient(seed):
    """With landmarks = all samples and full rank, Nystrom ROVER reduces to dense ROVER."""
    feats, nxt, xy, actions = make_data(seed=seed)
    model, kernel, gram, alpha = build(feats, nxt, xy, actions, rank=None)
    coeff = 0.3 * torch.randn(len(feats) + 1, 1, dtype=DTYPE, generator=torch.Generator().manual_seed(seed))
    pi = model.policy(coeff)
    reference = dense_reference(kernel, feats, nxt, actions, gram, alpha, pi)
    torch.testing.assert_close(model.loss(pi), reference.loss, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(model.gradient_coefficients(pi), reference.gradient_coefficients(),
                               rtol=1e-3, atol=1e-5)


def test_policy_is_a_distribution_and_uniform_at_zero_coefficients():
    feats, nxt, xy, actions = make_data(seed=2)
    model, *_ = build(feats, nxt, xy, actions)
    pi = model.policy(torch.zeros(len(feats) + 1, 1, dtype=DTYPE))
    torch.testing.assert_close(pi, torch.full_like(pi, 1.0 / pi.shape[1]))


def test_reduced_rank_gradient_error_decreases_with_rank():
    """The truncation discards eigendirections of the landmark Gram, so error shrinks as rank grows.

    Only 6 distinct states -> each action block has numerical rank <= 6, i.e. at most 24 useful
    directions; beyond that only the (ridge-regularised) null space is added.
    """
    feats, nxt, xy, actions = make_data(n=80, seed=3)
    index = torch.arange(80) % 6
    feats, xy = feats[index], xy[index]
    full, *_ = build(feats, nxt, xy, actions, rank=None)
    coeff = 0.2 * torch.randn(81, 1, dtype=DTYPE, generator=torch.Generator().manual_seed(3))
    pi = full.policy(coeff)
    reference = full.gradient_coefficients(pi)
    errors = {}
    for rank in (3, 8, 24, 80):
        reduced, *_ = build(feats, nxt, xy, actions, rank=rank)
        errors[rank] = float((reduced.gradient_coefficients(pi) - reference).norm() / reference.norm())
    assert errors[3] > errors[8] > errors[24] > errors[80], errors
    assert errors[24] < 0.05 and errors[80] < 1e-9, errors


def test_pmd_step_reduces_the_objective():
    feats, nxt, xy, actions = make_data(seed=4)
    model, *_ = build(feats, nxt, xy, actions)
    coeff = torch.zeros(len(feats) + 1, 1, dtype=DTYPE)
    pi = model.policy(coeff)
    loss = model.loss(pi)
    gradient = model.gradient_coefficients(pi)
    for eta in (1e-1, 1e-2, 1e-3, 1e-4):
        if model.loss(model.policy(coeff + eta * gradient)) < loss:
            return
    pytest.fail("no step size along the dual gradient decreased the objective")


# ----------------------------------------------------------------- pivoted Cholesky
def brute_force_pivots(kernel, feats, actions, size, force_first):
    """Reference greedy pivoted Cholesky on explicit Gram matrices (per-action blocks)."""
    same = (actions[:, None] == actions[None, :]).to(feats.dtype)
    K = kernel(feats, feats) * same
    residual = K.diagonal().clone()
    chosen = []
    for step in range(size):
        if step == 0 and force_first:
            pivot = 0
        else:
            masked = residual.clone()
            masked[chosen] = -float("inf")
            pivot = int(masked.argmax())
        chosen.append(pivot)
        S = chosen
        K_SS = K[S][:, S] + 1e-12 * torch.eye(len(S), dtype=feats.dtype)
        residual = K.diagonal() - (K[:, S] @ torch.linalg.solve(K_SS, K[S, :])).diagonal()
    return chosen


@pytest.mark.parametrize("kind", ["gaussian", "inner_product"])
@pytest.mark.parametrize("force_first", [False, True])
def test_pivoted_cholesky_matches_brute_force(kind, force_first):
    feats, _, _, actions = make_data(n=40, seed=5)
    kernel = Kernel(kind, 0.15 if kind == "gaussian" else None)
    got = pivoted_cholesky(feats, actions, 15, kernel, tolerance=0.0, force_first=force_first).tolist()
    want = brute_force_pivots(kernel, feats, actions, 15, force_first)
    assert got == want


def test_pivoted_cholesky_respects_tolerance_and_selects_distinct_rows():
    feats = torch.zeros(30, 3, dtype=DTYPE)
    feats[:, 0] = 1.0                                   # identical rows: rank-1 Gram
    actions = torch.zeros(30, dtype=torch.long)
    chosen = pivoted_cholesky(feats, actions, 10, Kernel("inner_product"), tolerance=1e-6)
    assert len(chosen) == 1
    feats, _, _, actions = make_data(n=40, seed=6)
    chosen = pivoted_cholesky(feats, actions, 40, Kernel("gaussian", 0.2))
    assert len(set(chosen.tolist())) == 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pivoted_cholesky_cpu_gpu_agree():
    feats, _, _, actions = make_data(n=300, seed=7)
    kernel = Kernel("gaussian", 0.2)
    cpu = pivoted_cholesky(feats, actions, 60, kernel, tolerance=1e-8)
    gpu = pivoted_cholesky(feats.cuda(), actions.cuda(), 60, kernel, tolerance=1e-8)
    assert cpu.tolist() == gpu.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nystrom_model_cpu_gpu_consistency():
    feats, nxt, xy, actions = make_data(n=64, seed=8)
    cpu_model, *_ = build(feats, nxt, xy, actions, rank=40)
    gpu_model, *_ = build(feats.cuda(), nxt.cuda(), xy.cuda(), actions.cuda(), rank=40)
    coeff = 0.2 * torch.randn(65, 1, dtype=DTYPE, generator=torch.Generator().manual_seed(8))
    pi = cpu_model.policy(coeff)
    torch.testing.assert_close(gpu_model.loss(pi.cuda()).cpu(), cpu_model.loss(pi), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(gpu_model.gradient_coefficients(pi.cuda()).cpu(),
                               cpu_model.gradient_coefficients(pi), rtol=1e-4, atol=1e-7)


def test_median_bandwidth_is_positive_and_scales():
    feats, *_ = make_data(seed=9)
    h = median_bandwidth(feats)
    assert h > 0
    assert median_bandwidth(feats, multiplier=2.0) == pytest.approx(2.0 * h)
