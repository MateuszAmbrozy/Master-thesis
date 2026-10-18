"""Run from any working directory; paths in config are project-relative."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from primal_mapf.runtime import read_config
from primal_mapf.training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classic or improved PRIMAL on CPU")
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Continue a checkpoint created by this trainer")
    group.add_argument("--init", help="Load model weights only, e.g. historical weights for fine-tuning")
    args = parser.parse_args()
    try:
        train(read_config(args.config), args.resume, args.init)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
