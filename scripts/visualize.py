"""Export a policy or solver trajectory as an SVG without plotting dependencies."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from primal_mapf.runtime import load_model, project_path, read_config, scenario
from primal_mapf.evaluation.evaluate import policy_episode
from primal_mapf.solvers.api import find_path
from primal_mapf.visualization import write_svg
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    algorithm = parser.add_mutually_exclusive_group(required=True)
    algorithm.add_argument("--variant", choices=["classic", "improved"])
    algorithm.add_argument("--solver", choices=["pibt", "ca", "cbs"])
    parser.add_argument("--output", default="results/trajectory.svg")
    parser.add_argument("--animate", action="store_true", help="Include SVG animation where supported")
    args = parser.parse_args()
    config = read_config(args.config)
    torch.set_num_threads(config.get("torch_threads", 1))
    config.update(config.get("scenarios", [{}])[0])
    obstacles, starts, goals = scenario(config, config["seed"])
    name = args.variant or args.solver
    if args.variant:
        model = load_model(args.variant, config["models"][args.variant])
        success, _, trace = policy_episode(model, args.variant, obstacles, starts, goals,
                                          config["max_steps"], config["seed"], config.get("greedy", False),
                                          return_trace=True, execution=config.get("execution", "joint"),
                                          profile=config.get("environment_profile", "common"))
    else:
        trace = find_path(args.solver, obstacles, starts, goals, config["max_steps"],
                          config.get("solver_timeout", 10), config["seed"])
        success = trace is not None
        trace = trace or [starts]
    output = project_path(args.output)
    write_svg(output, obstacles, starts, goals, trace, title=f"{name}: success={success}", animate=args.animate)
    print(output)


if __name__ == "__main__":
    main()
