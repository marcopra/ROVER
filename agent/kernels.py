"""Kernels used by ROVER on (learned or raw) features, with median-heuristic bandwidths."""

import torch


def squared_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, clamped at zero against round-off."""
    sq = (X * X).sum(1, keepdim=True) + (Y * Y).sum(1, keepdim=True).T - 2.0 * (X @ Y.T)
    return sq.clamp_min(0.0)


class Kernel:
    """``inner_product``: <x, y>.  ``gaussian``: exp(-||x - y||^2 / (2 h^2)) with bandwidth ``h``."""

    def __init__(self, kind: str, bandwidth: float | None = None):
        if kind not in ("inner_product", "gaussian"):
            raise ValueError(f"unknown kernel {kind!r}; expected 'inner_product' or 'gaussian'")
        self.kind = kind
        self.bandwidth = bandwidth

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.kind == "inner_product":
            return X @ Y.T
        if self.bandwidth is None or self.bandwidth <= 0.0:
            raise ValueError("gaussian kernel needs a positive bandwidth; fit one with median_bandwidth first")
        return torch.exp(-squared_distances(X, Y) / (2.0 * self.bandwidth ** 2))


def median_bandwidth(features: torch.Tensor, multiplier: float = 1.0, max_points: int = 1000) -> float:
    """Median pairwise distance of (a random subset of) ``features`` times ``multiplier``."""
    assert multiplier > 0.0
    features = features.detach().reshape(features.shape[0], -1).float()
    if features.shape[0] > max_points:
        features = features[torch.randperm(features.shape[0], device=features.device)[:max_points]]
    distances = torch.sqrt(squared_distances(features, features))
    distances = distances[distances > 0.0]
    median = 1.0 if distances.numel() == 0 else float(distances.median())
    return max(median * multiplier, 1e-12)
