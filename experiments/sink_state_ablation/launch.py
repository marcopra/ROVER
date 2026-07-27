from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


def quote_command(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/sink_state_ablation.yaml")
    parser.add_argument("--stage", choices=("pretrain", "adapt"), required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["output_dir"])
    commands = []
    for sink in cfg["sinks"]:
        for batch in sink["operator_batches"]:
            condition_id = f'{sink["label"]}__b{batch}'
            condition_dir = root / "runs" / condition_id
            if args.stage == "pretrain":
                seed = cfg["pretrain_seed"]
                parts = [
                    "conda", "run", "-n", "dist_matching", "python", "pretrain.py",
                    "--config-name", "scripts/experiment3/sink_ablation_pretrain",
                    f"seed={seed}",
                    f"agent.batch_size_actor={batch}",
                    f"agent.sink_schedule={sink['schedule']}",
                    f"ablation_run_dir={condition_dir.resolve()}",
                ]
                commands.append(quote_command(parts))
            else:
                checkpoint = condition_dir / "checkpoint" / "snapshot_50000.pt"
                for seed in cfg["adaptation_seeds"]:
                    adaptation_dir = condition_dir / f"adapt_s{seed}"
                    metrics = adaptation_dir / "adaptation.jsonl"
                    parts = [
                        "env", f"ROVER_ABLATION_RUN_DIR={adaptation_dir.resolve()}",
                        "conda", "run", "-n", "dist_matching", "python", "train.py",
                        "--config-name", "scripts/experiment3/sink_ablation_adaptation",
                        f"seed={seed}", f"p_path={checkpoint.resolve()}",
                        f"matched_metrics_path={metrics.resolve()}",
                        f"env.goal_position={cfg['downstream']['goal_position']}",
                    ]
                    commands.append(quote_command(parts))
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"launch_{args.stage}.sh"
    output.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    output.chmod(0o755)
    print(f"Wrote {len(commands)} commands to {output}")


if __name__ == "__main__":
    main()
