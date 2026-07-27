from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trapezoid_auc(frames, values, horizon):
    if not frames:
        return 0.0
    x = np.asarray(frames, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    x = np.concatenate(([0.0], x, [float(horizon)]))
    y = np.concatenate(([y[0]], y, [y[-1]]))
    keep = (x >= 0) & (x <= horizon)
    return float(np.trapz(y[keep], x[keep]) / horizon)


def largest_visitation_regression(coverage_rows):
    largest = 0.0
    for previous, current in zip(coverage_rows, coverage_rows[1:]):
        p = previous["visitation_distribution"]
        q = current["visitation_distribution"]
        largest = max(largest, max(
            (p[state] - q.get(state, 0.0) for state in p),
            default=0.0,
        ))
    return float(largest)


def summarize_pretrain(rows):
    coverage = sorted(
        (row for row in rows if row["event"] == "coverage"),
        key=lambda row: row["frame"],
    )
    operators = sorted(
        (row for row in rows if row["event"] == "operator_update"),
        key=lambda row: row["frame"],
    )
    if not coverage:
        raise ValueError("No coverage rows")
    final = coverage[-1]
    return {
        "final_policy_covered_states": final["covered_states"],
        "final_policy_coverage_fraction": final["coverage_fraction"],
        "largest_visitation_regression": largest_visitation_regression(coverage),
        "mean_estimated_sink_mass": float(np.mean([
            row["estimated_sink_mass"] for row in operators
            if "estimated_sink_mass" in row
        ])) if operators else None,
        "mean_low_support_query_fraction": float(np.mean([
            row["low_support_query_fraction"] for row in operators
            if "low_support_query_fraction" in row
        ])) if operators else None,
        "coverage_curve": [
            {
                "frame": row["frame"],
                "optimization_round": row["optimization_round"],
                "coverage_fraction": row["coverage_fraction"],
            }
            for row in coverage
        ],
    }


def summarize_adaptation(rows, horizon=60000):
    evaluations = sorted(
        (row for row in rows if row["event"] == "evaluation"),
        key=lambda row: row["frame"],
    )
    rewards = [row for row in rows if row["event"] == "first_reward"]
    return {
        "time_to_first_downstream_reward": (
            min(row["interactions"] for row in rewards) if rewards else None
        ),
        "downstream_learning_auc": trapezoid_auc(
            [row["frame"] for row in evaluations],
            [row["success_rate"] for row in evaluations],
            horizon,
        ),
        "downstream_curve": [
            {"frame": row["frame"], "success_rate": row["success_rate"]}
            for row in evaluations
        ],
    }
