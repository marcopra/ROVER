from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Emit frozen-pair adaptation commands.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--characterization", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    frozen = json.loads(Path(args.pairs).read_text())
    if frozen["selection_status"] != "matched":
        raise SystemExit("No eligible frozen pairs. Do not run downstream selection.")
    chars = json.loads(Path(args.characterization).read_text())["candidates"]
    candidates = {c["candidate_id"]: c for c in chars}
    commands = []
    for pair in frozen["pairs"]:
        for candidate_id in (pair["candidate_a"], pair["candidate_b"]):
            candidate = candidates[candidate_id]
            for goal in cfg["downstream"]["goals"]:
                for seed in cfg["downstream"]["seeds"]:
                    run_id = f'{pair["pair_id"]}__{candidate_id}__{goal["label"]}__s{seed}'
                    metrics = Path(cfg["output_dir"]) / "runs" / run_id / "metrics.jsonl"
                    goal_value = f'[{goal["position"][0]},{goal["position"][1]}]'
                    command = [
                        "conda", "run", "-n", "dist_matching", "python", "train.py",
                        "--config-name", "scripts/experiment1/multirooms_adaptation",
                        f'p_path={candidate["checkpoint"]}',
                        f'env.goal_position={goal_value}',
                        f"seed={seed}",
                        f"matched_metrics_path={metrics}",
                        f'hydra.run.dir={metrics.parent}',
                    ]
                    commands.append(" ".join(shlex.quote(part) for part in command))
    launch_file = Path(cfg["output_dir"]) / "launch_commands.sh"
    launch_file.parent.mkdir(parents=True, exist_ok=True)
    launch_file.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    launch_file.chmod(0o755)
    print(f"Wrote {len(commands)} commands to {launch_file}")
    if args.execute:
        import subprocess
        for command in commands:
            subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()

