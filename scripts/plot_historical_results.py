"""Regenerate archived result figures with English labels, using CSV data."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot(csv_path, destination, metric):
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        parameter = reader.fieldnames[0]
        rows = list(reader)
    key = lambda value: float(value.strip("()").split(",")[0])
    values = sorted({row[parameter] for row in rows}, key=key)
    algorithms = sorted({row["Algorithm"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        success = row["Success"].lower() in {"true", "1"}
        if metric == "success_rate":
            value = float(success) * 100
        elif metric == "time":
            value = float(row["Time"])
        elif success:
            value = float(row["Makespan"])
        else:
            continue
        grouped[row["Algorithm"], row[parameter]].append(value)
    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm in algorithms:
        means = [np.mean(grouped[algorithm, value]) if grouped[algorithm, value] else np.nan for value in values]
        ax.plot(range(len(values)), means, marker="o", label=algorithm.replace("_", " "))
    xlabel = {"agents": "Number of agents", "size": "Map size", "density": "Obstacle density"}[parameter]
    ylabel = {"success_rate": "Success rate (%)", "time": "Mean computation time (s)",
              "makespan": "Mean makespan (successful attempts only)"}[metric]
    ax.set(title=f"{ylabel} by {xlabel.lower()}", xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(range(len(values)), values)
    if metric == "success_rate":
        ax.set_ylim(-5, 105)
    if metric == "time" and "fixed_maps_base_vs_fine_tuned" not in str(csv_path):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Algorithm", loc="best")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT/"research/historical/file_manifest.json"))
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    records = json.loads(manifest_path.read_text())
    count = 0
    for record in records:
        path = ROOT/record["destination"]
        if path.suffix != ".png" or "historical/results" not in path.as_posix():
            continue
        metric = next(value for value in ["success_rate", "makespan", "time"] if path.stem.endswith("_"+value))
        stem = path.stem[:-(len(metric)+1)]
        candidates = [path.parent/("raw_"+stem+".csv"), path.parent.parent/("raw_"+stem+".csv")]
        csv_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if csv_path is None:
            raise FileNotFoundError(f"No source CSV for {path}")
        plot(csv_path, path, metric)
        record["destination_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["transformation"] = "English figure regenerated from archived measurements"
        record["data_file"] = csv_path.relative_to(ROOT).as_posix()
        count += 1
    manifest_path.write_text(json.dumps(records, indent=2)+"\n")
    print(f"Regenerated {count} historical figures with English labels.")


if __name__ == "__main__":
    main()
