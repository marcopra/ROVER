from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def state_ids(observations: np.ndarray) -> np.ndarray:
    """Convert scalar or one-hot observations to integer state ids."""
    x = np.asarray(observations)
    if x.ndim == 0:
        return x.reshape(1).astype(np.int64)
    if x.ndim == 1:
        if x.size > 1 and np.all((x == 0) | (x == 1)) and np.isclose(x.sum(), 1):
            return np.asarray([int(x.argmax())])
        return x.astype(np.int64)
    return x.reshape(x.shape[0], -1).argmax(axis=1).astype(np.int64)


def distribution(ids: Iterable[int], n_states: int, weights=None) -> np.ndarray:
    ids = np.asarray(list(ids), dtype=np.int64)
    counts = np.bincount(ids, weights=weights, minlength=n_states).astype(np.float64)
    return counts / counts.sum() if counts.sum() else counts


def coverage_metrics(ids: Iterable[int], n_states: int, weights=None) -> dict:
    p = distribution(ids, n_states, weights)
    support = int(np.count_nonzero(p))
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "support_count": support,
        "support_fraction": support / n_states,
        "entropy": entropy,
        "effective_support": float(np.exp(entropy)),
        "distribution": p.tolist(),
    }


def discounted_occupancy(trajectories: Iterable[Iterable[int]], n_states: int, gamma: float) -> dict:
    ids, weights = [], []
    for trajectory in trajectories:
        for t, state in enumerate(trajectory):
            ids.append(int(state))
            weights.append((1.0 - gamma) * gamma**t)
    return coverage_metrics(ids, n_states, weights)


def js_divergence(p, q) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _pair_key(a: Mapping, b: Mapping) -> tuple:
    return tuple(sorted((str(a["candidate_id"]), str(b["candidate_id"]))))


def match_candidates(candidates: list[dict], rule: Mapping) -> list[dict]:
    """Match using pretraining diagnostics only. Downstream fields are rejected."""
    forbidden = {"return", "success", "auc", "first_reward", "adaptation"}
    if any(any(token in key.lower() for token in forbidden) for c in candidates for key in c):
        raise ValueError("Candidate table contains downstream fields; pair selection must be blind.")
    max_gap = float(rule["max_buffer_support_gap"])
    min_final_gap = float(rule["min_final_effective_support_gap"])
    require_cross = bool(rule.get("require_cross_algorithm", True))
    eligible = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            if require_cross and a["algorithm"] == b["algorithm"]:
                continue
            bgap = abs(a["buffer"]["support_fraction"] - b["buffer"]["support_fraction"])
            fgap = abs(
                a["final_policy"]["effective_support"] - b["final_policy"]["effective_support"]
            ) / a["n_states"]
            if bgap <= max_gap and fgap >= min_final_gap:
                eligible.append(
                    {
                        "pair_id": "__".join(_pair_key(a, b)),
                        "candidate_a": a["candidate_id"],
                        "candidate_b": b["candidate_id"],
                        "buffer_support_gap": bgap,
                        "final_effective_support_gap": fgap,
                        "final_js_divergence": js_divergence(
                            a["final_policy"]["distribution"],
                            b["final_policy"]["distribution"],
                        ),
                    }
                )
    eligible.sort(
        key=lambda x: (
            -x["final_effective_support_gap"],
            -x["final_js_divergence"],
            x["buffer_support_gap"],
            x["pair_id"],
        )
    )
    used, selected = set(), []
    for pair in eligible:
        names = {pair["candidate_a"], pair["candidate_b"]}
        if not names & used:
            selected.append(pair)
            used |= names
    return selected[: int(rule.get("max_pairs", 5))]


def deterministic_subsample(ids: np.ndarray, size: int, seed: int) -> np.ndarray:
    ids = np.asarray(ids)
    if size > len(ids):
        raise ValueError("Subsample size exceeds buffer.")
    return ids[np.random.default_rng(seed).choice(len(ids), size=size, replace=False)]


def auc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def threshold_interactions(x, y, threshold: float):
    for step, value in zip(x, y):
        if value >= threshold:
            return int(step)
    return None


def bootstrap_ci(values, statistic=np.mean, confidence=0.95, samples=10_000, seed=0):
    values = np.asarray(values, float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.choice(values, len(values), replace=True)) for _ in range(samples)]
    alpha = (1 - confidence) / 2
    return [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))]


def config_digest(config: Mapping) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
