"""Gymnasium adapter exposing one joint action and one joint observation."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import MAPFEnv, PROFILES
from .runtime import scenario


class JointMAPFEnv(gym.Env):
    """Multi-agent observation with summed scalar reward and per-agent info.

    reset(options={"obstacles": ..., "starts": ..., "goals": ...}) installs
    a fixed scenario. Otherwise a scenario is sampled with Gymnasium's RNG.
    """
    metadata = {"render_modes": []}

    def __init__(self, agents=2, size=(8, 8), density=0.1, max_steps=256, profile="common"):
        super().__init__()
        from .config import validate_scenario, integer
        self.config = dict(agents=agents, size=list(size), density=density)
        validate_scenario(self.config)
        integer({"max_steps": max_steps}, "max_steps")
        if profile not in PROFILES:
            raise ValueError("Unknown environment profile")
        self.max_steps = max_steps
        self.env = MAPFEnv(num_agents=agents, settings=PROFILES[profile])
        self.action_space = spaces.MultiDiscrete([5] * agents)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, (agents, 4, 10, 10), dtype=np.float32),
            "goal": spaces.Box(-np.finfo(np.float32).max, np.finfo(np.float32).max, (agents, 3), dtype=np.float32),
        })
        self.steps = 0
        self._finished = True

    def _observation(self):
        observations = [self.env.get_obs(i + 1) for i in range(self.env.num_agents)]
        return {"grid": np.asarray([o[0] for o in observations], dtype=np.float32),
                "goal": np.asarray([o[1] for o in observations], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if options:
            if set(options) != {"obstacles", "starts", "goals"}:
                raise ValueError("Reset options must provide obstacles, starts, and goals")
            data = options["obstacles"], options["starts"], options["goals"]
        else:
            data = scenario(self.config, int(self.np_random.integers(0, 2**31)))
        self.env.set_custom_map(*data)
        self.steps = 0
        self._finished = False
        return self._observation(), {}

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must contain one integer in 0..4 per agent")
        result = self.env.step_joint(list(action))
        self.steps += 1
        truncated = self.steps >= self.max_steps and not result.done
        self._finished = result.done or truncated
        return self._observation(), float(sum(result.rewards)), result.done, truncated, {
            "agent_rewards": np.asarray(result.rewards, dtype=np.float32),
            "executed": np.asarray(result.executed, dtype=bool),
        }
