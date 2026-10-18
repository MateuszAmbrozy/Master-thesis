"""Validate public configuration inputs before starting expensive workflows."""
import math
from numbers import Real

VARIANTS = {"classic", "improved"}
SOLVERS = {"pibt", "ca", "cbs"}


def integer(config: dict, key: str, minimum: int = 1) -> None:
    value = config.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")


def number(config: dict, key: str, minimum: float = 0, maximum=None,
           strictly_positive: bool = False) -> None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if value < minimum or (strictly_positive and value == minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{key} is outside its supported range")


def validate_scenario(config: dict) -> None:
    integer(config, "agents")
    if config.get("map") is not None and (not isinstance(config["map"], str) or not config["map"]):
        raise ValueError("map must be a non-empty path string")
    size = config.get("size", [15, 15])
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError("size must contain two positive integers: [height, width]")
    density = config.get("density", 0.1)
    number({"density": density}, "density", maximum=1)
    if density == 1:
        raise ValueError("density must be less than 1")
    if not config.get("map") and size[0] * size[1] < 2 * config["agents"]:
        raise ValueError("Scenario needs at least twice as many free cells as agents")


def validate_config(config: dict, mode: str) -> None:
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    for key in ("episodes", "max_steps"):
        integer(config, key)
    integer(config, "seed", 0)
    integer({"torch_threads": config.get("torch_threads", 1)}, "torch_threads")
    if not isinstance(config.get("output"), str) or not config["output"].strip():
        raise ValueError("output must be a non-empty path string")
    if not isinstance(config.get("execution", "joint"), str) or config.get("execution", "joint") not in {"joint", "sequential"}:
        raise ValueError("execution must be 'joint' or 'sequential'")
    if not isinstance(config.get("environment_profile", "common"), str) or config.get("environment_profile", "common") not in {"common", "classic", "improved", "historical"}:
        raise ValueError("Unknown environment_profile")
    validate_scenario(config)
    if mode == "train":
        if not isinstance(config.get("variant"), str) or config.get("variant") not in VARIANTS:
            raise ValueError("variant must be 'classic' or 'improved'")
        for key in ("update_steps", "save_every"):
            integer(config, key)
        for key in ("learning_rate", "grad_clip"):
            number(config, key, strictly_positive=True)
        for key in ("gamma", "demonstration_prob"):
            number(config, key, maximum=1)
        for key in ("entropy_beta", "value_beta", "valid_beta", "blocking_beta"):
            number(config, key)
        number({"success_reward": config.get("success_reward", 20)}, "success_reward")
        if not isinstance(config.get("expert"), str) or config.get("expert") not in SOLVERS:
            raise ValueError("expert must be pibt, ca, or cbs")
        number({"expert_timeout": config.get("expert_timeout", 10)}, "expert_timeout", strictly_positive=True)
    elif mode == "evaluate":
        models, solvers = config.get("models", {}), config.get("solvers", [])
        if not isinstance(models, dict) or not set(models) <= VARIANTS:
            raise ValueError("models must map classic/improved to checkpoint paths")
        if any(not isinstance(p, str) or not p for p in models.values()):
            raise ValueError("Model checkpoint paths must be non-empty strings")
        if not isinstance(solvers, list) or any(not isinstance(s, str) or s not in SOLVERS for s in solvers) or len(set(solvers)) != len(solvers):
            raise ValueError("solvers must be a list of distinct supported solver names")
        if not models and not solvers:
            raise ValueError("Select at least one model or solver")
        if type(config.get("greedy", False)) is not bool:
            raise ValueError("greedy must be a boolean")
        number({"solver_timeout": config.get("solver_timeout", 10)}, "solver_timeout", strictly_positive=True)
        settings = config.get("scenarios", [{}])
        if not isinstance(settings, list) or not settings:
            raise ValueError("scenarios must be a non-empty list")
        for setting in settings:
            if not isinstance(setting, dict) or not set(setting) <= {"agents", "size", "density", "map"}:
                raise ValueError("Scenario overrides may contain only agents, size, density, and map")
            validate_scenario(dict(config, **setting))
    else:
        raise ValueError(f"Unknown configuration mode: {mode}")
