"""Nystrom landmark selection by pivoted Cholesky on the device.

ROVER's state-action kernel is ``k((x, a), (x', a')) = k(x, x') 1[a = a']``, so its Gram matrix
is block diagonal in the action.  Greedy pivoted Cholesky therefore keeps one growing Cholesky
factor per action; a pivot only updates the residual diagonal of its own block.  Everything stays
on the compute device, with a single host synchronisation per pivot (to read the arg-max).
"""

import numpy as np
import torch

from agent.kernels import Kernel


@torch.no_grad()
def pivoted_cholesky(
    features: torch.Tensor,
    actions: torch.Tensor,
    size: int,
    kernel: Kernel,
    tolerance: float = 0.0,
    force_first: bool = False,
) -> torch.Tensor:
    """Indices of ``size`` pivots chosen greedily by largest Cholesky residual.

    Args:
        features: (N, D) state features on the compute device.
        actions: (N,) integer actions; rows with different actions are uncorrelated.
        kernel: ``inner_product`` or ``gaussian`` kernel with a fitted bandwidth.
        tolerance: stop once the largest residual falls below ``tolerance`` times the
            largest initial residual (0 disables early stopping).
        force_first: always select row 0 first (the pinned reset transition).
    """
    assert features.ndim == 2 and features.is_floating_point() and torch.isfinite(features).all()
    assert actions.shape == (features.shape[0],) and tolerance >= 0.0
    count = features.shape[0]
    target = min(int(size), count)
    if target <= 0:
        return torch.empty(0, dtype=torch.long)

    if kernel.kind == "inner_product":
        residual = (features * features).sum(1)
    else:
        assert kernel.bandwidth is not None and kernel.bandwidth > 0.0
        residual = torch.ones(count, device=features.device, dtype=features.dtype)
    stop_threshold = tolerance * max(float(residual.max()), torch.finfo(residual.dtype).eps)

    # Rows of each action (original order preserved) and, for every row, (group, position in group).
    group_rows = [torch.nonzero(actions == a, as_tuple=True)[0] for a in torch.unique(actions)]
    routing = torch.empty((count, 2), dtype=torch.long)
    for group, rows in enumerate(group_rows):
        routing[rows.cpu(), 0] = group
        routing[rows.cpu(), 1] = torch.arange(len(rows))
    blocks = [features.index_select(0, rows) for rows in group_rows]
    factors = [features.new_empty((0, len(rows))) for rows in group_rows]
    ranks = [0] * len(group_rows)
    available = torch.ones(count, dtype=torch.bool, device=features.device)

    selected = []
    for step in range(target):
        if step == 0 and force_first:
            pivot = 0
            pivot_residual = float(residual[0])
        else:
            value, index = residual.masked_fill(~available, -torch.inf).max(dim=0)
            pivot_residual, pivot = torch.stack((value.double(), index.double())).tolist()  # one sync
            pivot = int(pivot)
            if tolerance > 0.0 and pivot_residual <= stop_threshold:
                break
        assert np.isfinite(pivot_residual), "non-finite Cholesky residual"
        pivot_residual = max(pivot_residual, 0.0)
        selected.append(pivot)
        available[pivot] = False
        if pivot_residual <= torch.finfo(residual.dtype).eps:
            # A (forced) pivot with zero kernel norm adds no direction; keep it as a landmark only.
            residual[pivot] = 0.0
            continue

        group, local = routing[pivot].tolist()
        block, rank, factor = blocks[group], ranks[group], factors[group]
        if rank == len(factor):  # grow geometrically
            grown = block.new_empty((min(target, len(block), max(32, 2 * rank)), len(block)))
            grown[:rank].copy_(factor)
            factors[group] = factor = grown
        if kernel.kind == "inner_product":
            column = block @ block[local]
        else:
            column = torch.exp(-(block - block[local]).square().sum(1) / (2.0 * kernel.bandwidth ** 2))
        if rank:
            column -= factor[:rank].T @ factor[:rank, local]
        factor[rank].copy_(column).div_(np.sqrt(pivot_residual))
        rows = group_rows[group]
        residual.index_copy_(0, rows, residual.index_select(0, rows).sub_(factor[rank].square()).clamp_(min=0.0))
        ranks[group] += 1
        residual[pivot] = 0.0

    assert torch.isfinite(residual).all(), "non-finite Cholesky residual"
    return torch.as_tensor(selected, dtype=torch.long)
