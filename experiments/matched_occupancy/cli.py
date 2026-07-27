from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .characterize import characterize
from .core import config_digest, match_candidates, write_json


def load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("characterize", "match"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.command == "characterize":
        result = [characterize(c, cfg) for c in cfg["candidates"]]
        write_json(args.output, {"config_digest": config_digest(cfg), "candidates": result})
    else:
        if not args.input:
            parser.error("match requires --input")
        data = json.loads(Path(args.input).read_text())
        pairs = match_candidates(data["candidates"], cfg["matching"])
        write_json(
            args.output,
            {
                "selection_status": "matched" if pairs else "no_eligible_pairs",
                "selection_blinded_to_downstream": True,
                "matching_rule": cfg["matching"],
                "config_digest": config_digest(cfg),
                "pairs": pairs,
            },
        )


if __name__ == "__main__":
    main()
