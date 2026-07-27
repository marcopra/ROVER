from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .core import read_jsonl, summarize_adaptation, summarize_pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/sink_state_ablation.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["output_dir"])
    results = []
    missing = []
    for sink in cfg["sinks"]:
        for batch in sink["operator_batches"]:
            condition_id = f'{sink["label"]}__b{batch}'
            condition_dir = root / "runs" / condition_id
            pretrain_rows = read_jsonl(condition_dir / "pretrain.jsonl")
            for seed in cfg["adaptation_seeds"]:
                run_id = f"{condition_id}__adapt_s{seed}"
                try:
                    row = {
                        "run_id": run_id, "sink": sink["label"],
                        "sink_schedule": sink["schedule"],
                        "operator_batch_size": batch,
                        "pretrain_seed": cfg["pretrain_seed"],
                        "seed": seed,
                        **summarize_pretrain(pretrain_rows),
                        **summarize_adaptation(
                            read_jsonl(condition_dir / f"adapt_s{seed}" / "adaptation.jsonl"),
                            cfg["downstream"]["auc_horizon"],
                        ),
                    }
                    results.append(row)
                except ValueError:
                    missing.append(run_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps({
        "status": "complete" if not missing else "incomplete",
        "paper_protocol": cfg["paper_protocol"],
        "per_seed": results,
        "support_interaction": support_interaction(results),
        "missing_runs": missing,
    }, indent=2))
    scalar_keys = [
        "run_id", "sink", "sink_schedule", "operator_batch_size",
        "pretrain_seed", "seed",
        "final_policy_covered_states", "final_policy_coverage_fraction",
        "largest_visitation_regression", "mean_estimated_sink_mass",
        "mean_low_support_query_fraction", "time_to_first_downstream_reward",
        "downstream_learning_auc",
    ]
    with (root / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in results)
    plot(results, root / "paper_plots.pdf")
    print(f"Wrote {len(results)} results; {len(missing)} missing")


def plot(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    metrics = [
        ("final_policy_coverage_fraction", "Final policy coverage"),
        ("largest_visitation_regression", "Largest visitation regression"),
        ("downstream_learning_auc", "Downstream success AUC"),
    ]
    labels = sorted({row["sink"] for row in results})
    batches = sorted({row["operator_batch_size"] for row in results})
    for ax, (metric, title) in zip(axes, metrics):
        for batch, marker in zip(batches, ("o", "s")):
            means = [
                np.mean([row[metric] for row in results
                         if row["sink"] == label
                         and row["operator_batch_size"] == batch
                         and row[metric] is not None])
                if any(row["sink"] == label and row["operator_batch_size"] == batch
                       and row[metric] is not None for row in results) else np.nan
                for label in labels
            ]
            ax.plot(range(len(labels)), means, marker=marker, label=f"batch {batch}")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.grid(alpha=.25)
    if batches:
        axes[0].legend(frameon=False)
    coverage_ax = axes[3]
    for batch, linestyle in zip(batches, ("-", "--")):
        for label in labels:
            curves = [
                row["coverage_curve"] for row in results
                if row["sink"] == label and row["operator_batch_size"] == batch
            ]
            if not curves:
                continue
            common_frames = sorted(set.intersection(*[
                {point["frame"] for point in curve} for curve in curves
            ]))
            means = [
                np.mean([
                    next(point["coverage_fraction"] for point in curve
                         if point["frame"] == frame)
                    for curve in curves
                ])
                for frame in common_frames
            ]
            coverage_ax.plot(
                common_frames, means, linestyle=linestyle,
                label=f"{label}, b={batch}",
            )
    coverage_ax.set_title("Coverage over optimization rounds")
    coverage_ax.set_xlabel("Environment frames")
    coverage_ax.set_ylabel("Coverage fraction")
    coverage_ax.grid(alpha=.25)
    coverage_ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def support_interaction(results):
    """Single-pretrain effect growth: partial-support effect minus main effect."""
    indexed = {}
    for row in results:
        indexed.setdefault((row["sink"], row["operator_batch_size"]), row)
    interactions = {}
    for sink in sorted({row["sink"] for row in results} - {"no_sink"}):
        keys = [
            (sink, 1024), ("no_sink", 1024),
            (sink, 5000), ("no_sink", 5000),
        ]
        estimates = []
        if all(key in indexed for key in keys):
            partial = (
                indexed[keys[0]]["final_policy_coverage_fraction"]
                - indexed[keys[1]]["final_policy_coverage_fraction"]
            )
            main = (
                indexed[keys[2]]["final_policy_coverage_fraction"]
                - indexed[keys[3]]["final_policy_coverage_fraction"]
            )
            estimates.append(partial - main)
        interactions[sink] = {
            "single_pretrain_estimates": estimates,
            "mean": float(np.mean(estimates)) if estimates else None,
            "standard_error": None,
        }
    return interactions


if __name__ == "__main__":
    main()
