"""Single-process recurrent actor-critic and behavioral cloning trainer.

This is a new, deterministic training entry point, not a reconstruction of
the asynchronous process schedule used to obtain historical checkpoints.
"""
import csv
import json
from typing import NamedTuple
from pathlib import Path

from .environment import MAPFEnv

import numpy as np
import torch
import torch.nn.functional as F

from .runtime import (communication, environment, forward, new_model,
                      project_path, scenario, seed_all, select_action, masked_policy, ROOT)
from .solvers.api import find_path
from .config import validate_config
from .provenance import runtime_metadata, file_sha256

TRAINER_VERSION = "primal_joint_masked_v2"
LOG_FIELDS = ["episode", "mode", "reward", "steps", "success", "loss"]


class EpisodeResult(NamedTuple):
    reward: float
    steps: int
    success: bool
    loss: float | None


def optimize(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.Tensor, clip: float) -> float:
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def rl_episode(model: torch.nn.Module, optimizer: torch.optim.Optimizer, variant: str, env: MAPFEnv, config: dict, rng: np.random.Generator) -> EpisodeResult:
    n = env.num_agents
    states = [model.init_state(1, "cpu") for _ in range(n)]
    previous = [0] * n
    buffers = [[] for _ in range(n)]
    reward_total, losses = 0.0, []
    done = env.world.done()
    steps = 0
    joint = config.get("execution", "joint") == "joint"
    while not done and steps < config["max_steps"]:
        steps += 1
        context = communication(variant, env, [s[0] for s in states])
        actions = [0] * n
        for i in (range(n) if joint else rng.permutation(n)):
            i = int(i)
            valid = env.legal_actions(i+1) if joint else env.listNextValidActions(i+1, previous[i])
            policy, value, h, c, blocking, _, policy_sig = forward(
                model, variant, env, i, *states[i], context)
            states[i] = h, c
            action = select_action(policy, valid, rng)
            previous[i] = action
            actions[i] = action
            if joint:
                reward, blocked = 0.0, False
            else:
                _, reward, done, _, _, blocked, _ = env.step((i+1, action))
            target_valid = torch.zeros(5)
            target_valid[valid] = 1
            p = masked_policy(policy, valid)[0]
            buffers[i].append((value.squeeze(), p[action].log(), -(p*p.clamp_min(1e-10).log()).sum(),
                               float(reward), policy_sig[0], target_valid,
                               blocking.squeeze(), float(blocked)))
            reward_total += reward
            if done:
                break
        if joint:
            transition = env.step_joint(actions)
            done = transition.done
            reward_total += sum(transition.rewards)
            for i in range(n):
                record = list(buffers[i][-1])
                record[3] = transition.rewards[i]
                record[7] = float(transition.blocked[i])
                buffers[i][-1] = tuple(record)
        if done:
            for buffer in buffers:
                if buffer:
                    record = list(buffer[-1])
                    record[3] += config.get("success_reward", 20.0)
                    buffer[-1] = tuple(record)
        # Flush the final partial rollout even when the time limit is reached.
        if done or steps % config["update_steps"] == 0 or steps == config["max_steps"]:
            context = communication(variant, env, [s[0] for s in states])
            loss = torch.zeros(())
            count = 0
            for i, buffer in enumerate(buffers):
                if not buffer:
                    continue
                bootstrap = 0.0
                if not done:
                    with torch.no_grad():
                        bootstrap = float(forward(model, variant, env, i, *states[i], context)[1].item())
                returns = []
                for record in reversed(buffer):
                    bootstrap = record[3] + config["gamma"] * bootstrap
                    returns.append(bootstrap)
                returns = torch.tensor(returns[::-1])
                values = torch.stack([r[0] for r in buffer])
                advantage = returns - values.detach()
                logps = torch.stack([r[1] for r in buffer])
                entropy = torch.stack([r[2] for r in buffer]).mean()
                valid_loss = F.binary_cross_entropy(torch.stack([r[4] for r in buffer]),
                                                    torch.stack([r[5] for r in buffer]))
                blocking_loss = F.binary_cross_entropy(torch.stack([r[6] for r in buffer]),
                                                       torch.tensor([r[7] for r in buffer]))
                loss = loss - (logps * advantage).mean() + config["value_beta"] * (returns-values).square().mean()
                loss = loss + config["valid_beta"] * valid_loss + config["blocking_beta"] * blocking_loss
                loss = loss - config["entropy_beta"] * entropy
                count += 1
            if count:
                losses.append(optimize(model, optimizer, loss/count, config["grad_clip"]))
            states = [(h.detach(), c.detach()) for h, c in states]
            buffers = [[] for _ in range(n)]
    return EpisodeResult(reward_total, steps, done, float(np.mean(losses)) if losses else None)


def imitation_episode(model: torch.nn.Module, optimizer: torch.optim.Optimizer, variant: str, env: MAPFEnv, config: dict, seed: int) -> EpisodeResult:
    path = find_path(config["expert"], env.getObstacleMap(), env.getAgents(), env.getGoals(),
                     max_steps=config["max_steps"], timeout=config.get("expert_timeout", 10), seed=seed)
    if path is None or len(path) < 2 or len(path)-1 > config["max_steps"]:
        return EpisodeResult(0.0, 0, False, None)
    # Replay the complete plan before updating; sequential mode rejects rotations.
    examples = []
    reward_total = 0.0
    for t in range(len(path)-1):
        observations = [env.get_obs(i+1) for i in range(env.num_agents)]
        positions = env.getAgents().copy()
        actions = [env.world.getAction((path[t+1][i][0]-path[t][i][0],
                                       path[t+1][i][1]-path[t][i][1])) for i in range(env.num_agents)]
        if config.get("execution", "joint") == "joint":
            transition = env.step_joint(actions)
            if not all(transition.executed) or env.getAgents() != list(map(tuple, path[t+1])):
                return EpisodeResult(reward_total, t, False, None)
            reward_total += sum(transition.rewards)
        else:
            pending = list(range(env.num_agents))
            while pending:
                progressed = False
                for i in pending.copy():
                    transition = env.step((i+1, actions[i]))
                    if transition.valid_action:
                        pending.remove(i)
                        reward_total += transition.reward
                        progressed = True
                if not progressed:
                    return EpisodeResult(reward_total, t, False, None)
        examples.append((observations, positions, actions))
    states = [model.init_state(1, "cpu") for _ in range(env.num_agents)]
    losses, chunk = [], []
    for t, (observations, positions, actions) in enumerate(examples):
        context = {}
        if variant == "improved":
            pos = torch.tensor(positions, dtype=torch.float32)
            visible = (pos[:, None]-pos[None, :]).abs().le(5).all(-1)
            context = {"h_neighbors": torch.stack([s[0].detach() for s in states], dim=1),
                       "comm_mask": (~visible)[None]}
        for i, (obs, goal) in enumerate(observations):
            kwargs = dict(context, agent_idx=i) if variant == "improved" else {}
            p, _, h, c, _, _, _ = model(torch.tensor(np.asarray(obs), dtype=torch.float32)[None],
                                       torch.tensor(goal, dtype=torch.float32)[None], *states[i], **kwargs)
            states[i] = h, c
            chunk.append(-p[0, actions[i]].clamp_min(1e-10).log())
        if (t+1) % config["update_steps"] == 0 or t == len(examples)-1:
            losses.append(optimize(model, optimizer, torch.stack(chunk).mean(), config["grad_clip"]))
            states = [(h.detach(), c.detach()) for h, c in states]
            chunk = []
    return EpisodeResult(reward_total, len(examples), env.world.done(), float(np.mean(losses)))


def train(config: dict, resume: str | Path | None = None, init: str | Path | None = None) -> Path | None:
    if resume and init:
        raise ValueError("Use either resume or init")
    validate_config(config, "train")
    torch.set_num_threads(config.get("torch_threads", 1))
    seed_all(config["seed"])
    variant = config["variant"]
    model = new_model(variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    episode = 0
    if resume or init:
        checkpoint = torch.load(project_path(resume or init), map_location="cpu", weights_only=True)
        if checkpoint.get("variant", variant) != variant:
            raise ValueError("Checkpoint variant mismatch")
        model.load_state_dict(checkpoint["model"])
        if resume:
            if checkpoint.get("trainer") != TRAINER_VERSION:
                raise ValueError("Different trainer version; use --init to start a new run")
            previous = checkpoint["config"]
            keys = set(config) | set(previous)
            if any(config.get(k) != previous.get(k) for k in keys - {"episodes", "output", "save_every"}):
                raise ValueError("Resume configuration differs from saved training configuration")
            optimizer.load_state_dict(checkpoint["optimizer"])
            episode = int(checkpoint["episode"])
    if episode > config["episodes"]:
        raise ValueError("episodes cannot be lower than the checkpoint episode")
    output = project_path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError("Output is not empty; use --resume or a new output directory")
    if resume:
        existing = output / "checkpoint.pt"
        if existing.exists() and existing.resolve() != project_path(resume).resolve() and file_sha256(existing) != file_sha256(project_path(resume)):
            raise FileExistsError("Resume output contains a different checkpoint; choose a new directory")
        reconcile_log(output / "training.csv", episode)
    metadata = runtime_metadata(ROOT)
    if resume or init:
        metadata["initial_checkpoint_sha256"] = file_sha256(project_path(resume or init))
        metadata["initialization_mode"] = "resume" if resume else "init"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")

    def save():
        destination = output/"checkpoint.pt"
        temporary = output/"checkpoint.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "episode": episode, "variant": variant, "config": config,
                    "trainer": TRAINER_VERSION, "metadata": metadata}, temporary)
        temporary.replace(destination)

    completed_episode = episode
    save()
    try:
        with (output/"training.csv").open("a", newline="") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(LOG_FIELDS)
            while episode < config["episodes"]:
                episode_seed = config["seed"] + episode
                seed_all(episode_seed)
                rng = np.random.default_rng(episode_seed)
                obstacles, starts, goals = scenario(config, episode_seed)
                env = environment(variant, obstacles, starts, goals, config.get("environment_profile", "common"))
                if rng.random() < config["demonstration_prob"]:
                    mode = "IL"
                    result = imitation_episode(model, optimizer, variant, env, config, episode_seed)
                else:
                    mode = "RL"
                    result = rl_episode(model, optimizer, variant, env, config, rng)
                episode += 1
                writer.writerow([episode, mode, *result])
                stream.flush()
                print(f"episode={episode} mode={mode} reward={result[0]:.2f} steps={result[1]} success={result[2]} loss={result[3]}", flush=True)
                if episode % config["save_every"] == 0:
                    save()
                    completed_episode = episode
    except KeyboardInterrupt:
        # Partial episodes may contain optimizer updates: do not label them
        # as a resumable completed episode. Keep the last consistent snapshot.
        print(f"Interrupted. Last periodic checkpoint: episode {completed_episode}.")
        return
    save()
    return output/"checkpoint.pt"


def reconcile_log(path, episode: int) -> None:
    """Atomically retain each committed episode once before a resumed run.

    Missing history is permitted when resuming into a fresh directory. Existing
    rows are validated before replacement; unrelated malformed logs are rejected.
    """
    if not path.exists():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LOG_FIELDS:
            raise ValueError("Training log has an unexpected header")
        committed = {}
        for row in reader:
            try:
                index = int(row["episode"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Training log contains an invalid episode") from exc
            if index < 1 or None in row or any(v is None for v in row.values()):
                raise ValueError("Training log contains an invalid row")
            if index <= episode:
                committed[index] = row
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(committed[k] for k in sorted(committed))
    temporary.replace(path)
