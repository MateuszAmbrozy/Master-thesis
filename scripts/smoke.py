"""Run training, evaluation, and SVG export without historical checkpoints."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from primal_mapf.runtime import ROOT, project_path, read_config
from primal_mapf.training import train
from primal_mapf.evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/quickstart")
    args = parser.parse_args()
    output = project_path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty smoke output directory")
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    for variant in ("classic", "improved"):
        config = read_config(ROOT / f"configs/smoke_{variant}.json")
        config["output"] = str(output / variant)
        models[variant] = str(train(config))
    config = read_config(ROOT / "configs/experiments/smoke.json")
    config.update(models=models, output=str(output / "evaluation"))
    config_path = output / "evaluation_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    evaluate(config)
    subprocess.run([sys.executable, str(ROOT / "scripts/visualize.py"), "--config", str(config_path),
                    "--variant", "classic", "--output", str(output / "trajectory.svg")], check=True)
    print(f"Smoke workflow completed: {output}")


if __name__ == "__main__":
    main()
