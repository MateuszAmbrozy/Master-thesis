"""Shared model interfaces and reproducible scenario construction."""
from __future__ import annotations
import json
import random
import os
from pathlib import Path

import numpy as np
import torch

from .types import NetworkOutput
from .environment import MAPFEnv

# Editable installs resolve the checkout; wheels use an explicit workspace or cwd.
_checkout = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("PRIMAL_MAPF_ROOT", _checkout if (_checkout / "configs").is_dir() else Path.cwd())).resolve()


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    # Paths inside configuration are relative to the project, never to cwd.
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    config.setdefault("seed", 42)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classes(variant: str) -> tuple[type[torch.nn.Module], type[MAPFEnv]]:
    if variant == "classic":
        from .classic.network import ACNet
        from .classic.environment import MAPFEnv
    elif variant == "improved":
        from .improved.network import ACNet
        from .improved.environment import MAPFEnv
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return ACNet, MAPFEnv


def new_model(variant: str) -> torch.nn.Module:
    model_cls, _ = classes(variant)
    return model_cls(fov_size=10)


def load_model(variant: str, path: str | Path) -> torch.nn.Module:
    model = new_model(variant)
    checkpoint = torch.load(project_path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("variant", variant) != variant:
        raise ValueError("Checkpoint variant does not match requested model")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_map(path: str | Path) -> np.ndarray:
    lines = project_path(path).read_text().splitlines()
    start = lines.index("map")
    height = int(next(line.split()[1] for line in lines[:start] if line.startswith("height ")))
    width = int(next(line.split()[1] for line in lines[:start] if line.startswith("width ")))
    rows = lines[start + 1:]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError("Invalid MovingAI map dimensions")
    return np.array([[cell not in '.G' for cell in row] for row in rows], dtype=int)


def scenario(config: dict, seed: int, agents: int | None = None) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate one scenario with distinct endpoints in a connected component."""
    from collections import deque
    rng = np.random.default_rng(seed)
    from .config import validate_scenario
    agents = config["agents"] if agents is None else agents
    validate_scenario(dict(config, agents=agents))
    size = config.get("size", [15, 15])
    if agents < 1 or (not config.get("map") and np.prod(size) < 2 * agents):
        raise ValueError("Scenario needs at least twice as many free cells as agents")
    density = config.get("density", 0.1)
    if not 0 <= density < 1:
        raise ValueError("Density must be in [0, 1)")
    for _ in range(200):
        obstacles = load_map(config["map"]) if config.get("map") else (rng.random(size) < density).astype(int)
        free = set(map(tuple, np.argwhere(obstacles == 0)))
        largest = []
        while free:
            start = min(free)
            free.remove(start)
            queue, component = deque([start]), [start]
            while queue:
                r, c = queue.popleft()
                for p in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if p in free:
                        free.remove(p)
                        component.append(p)
                        queue.append(p)
            if len(component) > len(largest):
                largest = component
        if len(largest) >= 2 * agents:
            selected = rng.choice(len(largest), 2 * agents, replace=False)
            endpoints = [tuple(map(int, largest[int(i)])) for i in selected]
            return obstacles, endpoints[:agents], endpoints[agents:]
        if config.get("map"):
            break
    raise ValueError("Cannot generate a sufficiently large connected scenario")


def environment(variant: str, obstacles: np.ndarray, starts: list, goals: list, profile: str = "historical") -> MAPFEnv:
    _, env_cls = classes(variant)
    from .environment import PROFILES
    name = variant if profile == "historical" else profile
    if name not in PROFILES:
        raise ValueError(f"Unknown environment profile: {profile}")
    env = env_cls(num_agents=len(starts), fov_size=10, settings=PROFILES[name])
    env.set_custom_map(obstacles.copy(), list(starts), list(goals))
    return env


def communication(variant: str, env: MAPFEnv, hidden: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if variant == "classic":
        return {}
    positions = torch.tensor(env.getAgents(), dtype=torch.float32)
    visible = (positions[:, None] - positions[None, :]).abs().le(5).all(-1)
    return {"h_neighbors": torch.stack([h.detach() for h in hidden], dim=1),
            "comm_mask": (~visible).unsqueeze(0)}


def forward(model: torch.nn.Module, variant: str, env: MAPFEnv, index: int, h: torch.Tensor, c: torch.Tensor, context: dict) -> NetworkOutput:
    obs, goal = env.get_obs(index + 1)
    kwargs = dict(context, agent_idx=index) if variant == "improved" else {}
    return model(torch.as_tensor(np.asarray(obs), dtype=torch.float32)[None],
                 torch.tensor(goal, dtype=torch.float32)[None], h, c, **kwargs)


def select_action(policy: torch.Tensor, valid: list[int], rng: np.random.Generator, greedy: bool = False) -> int:
    probabilities = masked_policy(policy, valid).detach().cpu().numpy().reshape(-1).copy()
    mask = np.zeros_like(probabilities)
    mask[valid] = 1
    probabilities *= mask
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 1e-12:
        return int(rng.choice(valid))
    probabilities /= total
    return int(probabilities.argmax()) if greedy else int(rng.choice(len(probabilities), p=probabilities))


def masked_policy(policy: torch.Tensor, valid: list[int]) -> torch.Tensor:
    """Normalize legal actions without detaching the policy gradient."""
    if not valid or any(a not in range(policy.shape[-1]) for a in valid):
        raise ValueError("At least one valid action index is required")
    if not torch.isfinite(policy).all() or (policy < 0).any():
        raise ValueError("Policy probabilities must be finite and non-negative")
    mask = torch.zeros_like(policy)
    mask[..., valid] = 1
    # A small legal-only floor handles underflow while keeping invalid mass zero.
    probabilities = policy.clamp_min(torch.finfo(policy.dtype).tiny) * mask
    return probabilities / probabilities.sum(dim=-1, keepdim=True)
