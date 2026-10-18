import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from primal_mapf.runtime import read_config
from primal_mapf.evaluation.evaluate import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare models and solvers on shared scenarios")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override the configured output directory")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output:
        config["output"] = args.output
    try:
        evaluate(config)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
