"""Train or evaluate PRIMAL on maze, room, and warehouse maps."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from primal_mapf.config import validate_config
from primal_mapf.fixed_maps import (MAPS, apply_overrides, evaluation_config,
                                    initial_weights, training_config)
from primal_mapf.runtime import project_path
from primal_mapf.training import train
from primal_mapf.evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--map", choices=MAPS, required=True)
        sub.add_argument("--variant", choices=["classic", "improved"] + (["both"] if command == "evaluate" else []),
                         required=command == "train", default="both" if command == "evaluate" else None)
        sub.add_argument("--episodes", type=int)
        sub.add_argument("--agents", type=int)
        sub.add_argument("--max-steps", type=int)
        sub.add_argument("--seed", type=int)
        sub.add_argument("--output")
        sub.add_argument("--execution", choices=["joint", "sequential"])
        sub.add_argument("--environment-profile", choices=["common", "historical", "classic", "improved"])
        sub.add_argument("--dry-run", action="store_true", help="Print the configuration without starting a run")
        if command == "train":
            source = sub.add_mutually_exclusive_group()
            source.add_argument("--init", help="base, fine-tuned, or a checkpoint path")
            source.add_argument("--resume", help="Resume using the configuration saved in this checkpoint")
            sub.add_argument("--expert", choices=["pibt", "ca", "cbs"])
        else:
            sub.add_argument("--weights", choices=["base", "fine-tuned", "trained", "scratch"], default="fine-tuned")
            sub.add_argument("--checkpoint", help="Override weights for a single selected variant")
            sub.add_argument("--solvers", nargs="*", choices=["pibt", "ca", "cbs"], help="Use --solvers alone to omit solvers")
    args = parser.parse_args()
    try:
        if args.command == "train":
            init = initial_weights(args.map, args.variant, args.init)
            if args.resume:
                saved = torch.load(project_path(args.resume), map_location="cpu", weights_only=True)
                config = saved.get("config")
                if not isinstance(config, dict):
                    raise ValueError("Checkpoint cannot resume; use --init instead")
                if config.get("variant") != args.variant or project_path(config.get("map", "")).resolve() != project_path(MAPS[args.map]).resolve():
                    raise ValueError("Resume checkpoint does not match the selected map and variant")
            else:
                config = training_config(args.map, args.variant, init)
            config = apply_overrides(config, args)
            if args.expert is not None:
                config["expert"] = args.expert
            validate_config(config, "train")
            if args.dry_run:
                print(json.dumps(dict(config=config, init=init, resume=args.resume), indent=2))
            else:
                train(config, init=init, resume=args.resume)
        else:
            config = apply_overrides(evaluation_config(args.map, args.weights, args.variant, args.checkpoint), args)
            if args.solvers is not None:
                config["solvers"] = args.solvers
            validate_config(config, "evaluate")
            if args.dry_run:
                print(json.dumps(config, indent=2))
            else:
                missing = [path for path in config["models"].values() if not project_path(path).is_file()]
                if missing:
                    raise FileNotFoundError("Missing model weights: " + ", ".join(missing) +
                                            ". Select --weights base, import historical weights, or train on this map first.")
                evaluate(config)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
