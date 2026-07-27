from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from .core import auc, bootstrap_ci, threshold_interactions, write_json


def read_runs(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("runs/*/metrics.jsonl")):
        parts = path.parent.name.split("__")
        if len(parts) < 5:
            continue
        pair_id, candidate, goal, seed = "__".join(parts[:-3]), parts[-3], parts[-2], parts[-1]
        for line in path.read_text().splitlines():
            rows.append({"pair_id": pair_id, "candidate_id": candidate, "goal": goal,
                         "seed": int(seed[1:]), **json.loads(line)})
    return pd.DataFrame(rows)


def correlations(table, x, y, bootstrap_samples, seed):
    clean = table[[x, y]].dropna()
    if len(clean) < 3:
        return {"n": len(clean), "pearson": None, "spearman": None, "bootstrap_ci": [None, None]}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstrap_samples):
        sample = clean.iloc[rng.integers(0, len(clean), len(clean))]
        if sample[x].nunique() > 1 and sample[y].nunique() > 1:
            boot.append(spearmanr(sample[x], sample[y]).statistic)
    return {
        "n": len(clean),
        "pearson": float(pearsonr(clean[x], clean[y]).statistic),
        "spearman": float(spearmanr(clean[x], clean[y]).statistic),
        "bootstrap_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--characterization", required=True)
    p.add_argument("--pairs", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    runs = read_runs(out)
    if runs.empty:
        raise SystemExit("No downstream metrics found; report remains NOT RUN.")
    evaluations = runs[runs.event == "evaluation"].copy()
    summaries = []
    for keys, group in evaluations.groupby(["candidate_id", "goal", "seed"]):
        group = group.sort_values("frame")
        summaries.append({
            "candidate_id": keys[0], "goal": keys[1], "seed": keys[2],
            "zero_shot_success": float(group.iloc[0].success_rate),
            "success_auc": auc(group.frame, group.success_rate),
            "threshold_interactions": threshold_interactions(
                group.frame, group.success_rate, cfg["analysis"]["success_threshold"]),
        })
    summary = pd.DataFrame(summaries)
    first = runs[runs.event == "first_reward"][["candidate_id", "goal", "seed", "interactions",
                                                 "before_first_update"]]
    summary = summary.merge(first, how="left", on=["candidate_id", "goal", "seed"])
    chars = json.loads(Path(args.characterization).read_text())["candidates"]
    features = pd.DataFrame([{
        "candidate_id": c["candidate_id"],
        "algorithm": c["algorithm"],
        "buffer_support": c["buffer"]["support_fraction"],
        "buffer_effective_support": c["buffer"]["effective_support"],
        "final_support": c["final_policy"]["support_fraction"],
        "final_effective_support": c["final_policy"]["effective_support"],
    } for c in chars])
    summary = summary.merge(features, on="candidate_id")
    summary.to_csv(out / "adaptation_summary.csv", index=False)
    aggregate = summary.groupby(["candidate_id", "algorithm"], as_index=False).agg(
        zero_shot_success=("zero_shot_success", "mean"),
        success_auc=("success_auc", "mean"),
        interactions_until_first_reward=("interactions", "mean"),
        threshold_interactions=("threshold_interactions", "mean"),
        buffer_support=("buffer_support", "first"),
        final_effective_support=("final_effective_support", "first"),
    )
    aggregate.to_csv(out / "matched_pair_table.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    axes[0].scatter(aggregate.buffer_support, aggregate.success_auc)
    axes[0].set(xlabel="Replay-buffer support", ylabel="Adaptation success AUC")
    axes[1].scatter(aggregate.final_effective_support, aggregate.success_auc)
    axes[1].set(xlabel="Final discounted effective support")
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_adaptation.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for (candidate, goal), group in evaluations.groupby(["candidate_id", "goal"]):
        curve = group.groupby("frame").success_rate.agg(["mean", "sem"]).reset_index()
        ax.plot(curve.frame, curve["mean"], label=f"{candidate}/{goal}")
        ax.fill_between(curve.frame, curve["mean"] - 1.96 * curve.sem,
                        curve["mean"] + 1.96 * curve.sem, alpha=.15)
    ax.set(xlabel="Interactions", ylabel="Success rate")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "per_goal_learning_curves.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for candidate, group in evaluations.groupby("candidate_id"):
        per_seed = group.groupby(["seed", "frame"], as_index=False).success_rate.mean()
        curve = per_seed.groupby("frame").success_rate.agg(["mean", "sem"]).reset_index()
        ax.plot(curve.frame, curve["mean"], label=candidate)
        ax.fill_between(curve.frame, curve["mean"] - 1.96 * curve.sem,
                        curve["mean"] + 1.96 * curve.sem, alpha=.15)
    ax.set(xlabel="Interactions", ylabel="Goal-aggregate success rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "aggregate_learning_curves.png", dpi=200)
    plt.close(fig)
    correlation = {
        feature: correlations(aggregate, feature, "success_auc",
                              cfg["analysis"]["bootstrap_samples"], cfg["analysis"]["bootstrap_seed"])
        for feature in ("buffer_support", "final_effective_support")
    }
    write_json(out / "results.json", {
        "status": "complete",
        "num_adaptation_seeds": len(cfg["downstream"]["seeds"]),
        "correlations_with_success_auc": correlation,
        "aggregate": aggregate.where(pd.notnull(aggregate), None).to_dict("records"),
    })


if __name__ == "__main__":
    main()
