"""Configuration selection for fixed-map training and evaluation."""
from .runtime import ROOT, project_path, read_config

MAPS = {
    "maze": "maps/maze-32-32-4.map",
    "room": "maps/room-64-64-16.map",
    "warehouse": "maps/warehouse-10-20-10-2-1.map",
}


def initial_weights(map_name: str, variant: str, value: str | None):
    if value == "base":
        return f"checkpoints/base/{variant}.pt"
    if value == "fine-tuned":
        return f"checkpoints/fine_tuned/{variant}/{map_name}.pt"
    return value


def training_config(map_name: str, variant: str, init=None) -> dict:
    mode = "finetune" if init else "train"
    return read_config(ROOT/f"configs/maps/{map_name}/{mode}_{variant}.json")


def evaluation_config(map_name: str, weights: str, variant="both", checkpoint=None) -> dict:
    config = read_config(ROOT/f"configs/maps/{map_name}/evaluate_{weights.replace('-', '_')}.json")
    if checkpoint and variant == "both":
        raise ValueError("--checkpoint requires --variant classic or improved")
    if variant != "both":
        config["models"] = {variant: checkpoint or config["models"][variant]}
        config["output"] += f"_{variant}"
    return config


def apply_overrides(config: dict, args) -> dict:
    config = dict(config)
    for key in ("episodes", "max_steps", "output", "seed", "execution", "environment_profile"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.agents is not None:
        config["agents"] = args.agents
        config.pop("scenarios", None)
    return config
