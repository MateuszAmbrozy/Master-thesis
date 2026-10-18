"""Evaluate policies and solvers using explicit transition semantics."""
import csv
import json
import math
import time

import numpy as np
import torch

from ..config import validate_config
from ..provenance import file_sha256, runtime_metadata
from ..runtime import (ROOT, communication, environment, forward, load_model,
                       project_path, scenario, seed_all, select_action)
from ..solvers.api import find_path, valid_plan


def policy_episode(model, variant, obstacles, starts, goals, max_steps, seed,
                   greedy=False, return_trace=False, execution="joint", profile="common"):
    """Roll out a policy; all joint decisions use one pre-transition state."""
    if execution not in {"joint", "sequential"}:
        raise ValueError("execution must be joint or sequential")
    env = environment(variant, obstacles, starts, goals, profile)
    rng = np.random.default_rng(seed)
    states = [model.init_state(1, "cpu") for _ in starts]
    previous = [0] * len(starts)
    steps = 0
    trace = [env.getAgents().copy()]
    with torch.inference_mode():
        while not env.world.done() and steps < max_steps:
            steps += 1
            context = communication(variant, env, [s[0] for s in states])
            actions = [0] * len(starts)
            order = range(len(starts)) if execution == "joint" else rng.permutation(len(starts))
            for index in order:
                i = int(index)
                result = forward(model, variant, env, i, *states[i], context)
                states[i] = result.hidden, result.cell
                legal = env.legal_actions(i+1) if execution == "joint" else env.listNextValidActions(i+1, previous[i])
                actions[i] = select_action(result.policy, legal, rng, greedy)
                previous[i] = actions[i]
                if execution == "sequential":
                    env.step((i+1, actions[i]))
                    if env.world.done():
                        break
            if execution == "joint":
                env.step_joint(actions)
            trace.append(env.getAgents().copy())
    # Validate failed trajectories too, with their actual terminal configuration.
    if not valid_plan(trace, obstacles, starts, trace[-1]):
        raise RuntimeError("Policy produced an invalid MAPF trajectory")
    result = env.world.done(), steps if env.world.done() else -1
    return (*result, trace) if return_trace else result


def summarize(rows: list[dict]) -> dict:
    """Report sample dispersion and a Wilson 95% interval for success rate."""
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    rate = successes / n
    z = 1.959963984540054
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
    times = [r["seconds"] for r in rows]
    makespans = [r["makespan"] for r in rows if r["success"]]
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": rate,
        "success_rate_ci95": [max(0., center-half), min(1., center+half)],
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times, ddof=1)) if n > 1 else None,
        "mean_makespan_successful": float(np.mean(makespans)) if makespans else None,
        "std_makespan_successful": float(np.std(makespans, ddof=1)) if len(makespans) > 1 else None,
    }


def evaluate(config: dict) -> dict:
    validate_config(config, "evaluate")
    torch.set_num_threads(config.get("torch_threads", 1))
    models = {variant: load_model(variant, path) for variant, path in config.get("models", {}).items()}
    names = [*models, *config.get("solvers", [])]
    execution = config.get("execution", "joint")
    profile = config.get("environment_profile", "common")
    output = project_path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output is not empty; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    metadata = runtime_metadata(ROOT)
    metadata["execution"] = execution
    metadata["environment_profile"] = profile
    metadata["model_sha256"] = {name: file_sha256(project_path(path)) for name, path in config.get("models", {}).items()}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    rows, scenarios = [], []
    fields = ["setting", "scenario", "seed", "agents", "height", "width", "density", "map", "algorithm", "execution", "environment_profile", "success", "makespan", "seconds"]
    with (output/"raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for setting_id, setting in enumerate(config.get("scenarios", [{}])):
            parameters = dict(config, **setting)
            for ep in range(config["episodes"]):
                sid = len(scenarios)
                seed = config["seed"] + sid
                seed_all(seed)
                obstacles, starts, goals = scenario(parameters, seed)
                scenarios.append({"setting": setting_id, "id": sid, "seed": seed,
                                  "obstacles": obstacles.tolist(), "starts": starts, "goals": goals})
                # Persist inputs before executing potentially interrupted solvers.
                temporary = output / "scenarios.tmp"
                temporary.write_text(json.dumps(scenarios))
                temporary.replace(output / "scenarios.json")
                for name in names:
                    seed_all(seed)
                    begin = time.perf_counter()
                    if name in models:
                        success, makespan = policy_episode(models[name], name, obstacles, starts, goals,
                                                           config["max_steps"], seed, config.get("greedy", False),
                                                           execution=execution, profile=profile)
                        semantics = execution
                    else:
                        path = find_path(name, obstacles, starts, goals, config["max_steps"],
                                         config.get("solver_timeout", 10), seed)
                        success = path is not None
                        makespan = len(path)-1 if success else -1
                        semantics = "joint"
                    row = dict(setting=setting_id, scenario=sid, seed=seed, agents=len(starts),
                               height=obstacles.shape[0], width=obstacles.shape[1],
                               density=float(obstacles.mean()), map=parameters.get("map", "random"), algorithm=name,
                               execution=semantics, environment_profile=profile if name in models else "not_applicable",
                               success=int(success), makespan=makespan, seconds=time.perf_counter()-begin)
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    print(row, flush=True)
    summary = {name: summarize([r for r in rows if r["algorithm"] == name]) for name in names}
    for name in names:
        summary[name]["execution"] = execution if name in models else "joint"
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (output/"summary_by_setting.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "algorithm", "execution", "episodes", "success_rate", "success_ci95_low", "success_ci95_high", "mean_seconds", "std_seconds", "mean_makespan_successful", "std_makespan_successful"])
        for sid in sorted({r["setting"] for r in rows}):
            for name in names:
                result = summarize([r for r in rows if r["algorithm"] == name and r["setting"] == sid])
                writer.writerow([sid, name, summary[name]["execution"], result["episodes"], result["success_rate"],
                                 *result["success_rate_ci95"], result["mean_seconds"], result["std_seconds"],
                                 result["mean_makespan_successful"], result["std_makespan_successful"]])
    return summary
