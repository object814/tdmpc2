"""
Custom Metaworld environment maker for TD-MPC2.

Mirrors the DreamerV3 `make_env` pipeline in
`third_party/dreamerv3/dreamer_sequential.py`:

  gymnasium.make("Meta-World/MT1", env_name=task)
    -> ProprioMultiImageObsWrapper (topview + front + gripperPOV stacked, 128x128)
    -> RewardTuningWrapperV2 ((-1,1) -> (-1,0))
        -> _Gymnasium5To4 (classic gym-style 4-tuple API for TD-MPC2 compatibility)
    -> _DictObsAdapter (rename keys to 'state'/'rgb', CHW layout)
    -> ActionRepeat (k=2, sum rewards; TD-MPC2 default behaviour)
    -> Timeout (configurable max_episode_steps)

Task naming: either `metaworld_<env_name>` (DreamerV3 style, preferred) or
`mw-<env_name>`. Example: `metaworld_drawer-open-v3` or `mw-drawer-open-v3`.
"""

import sys
from pathlib import Path

import gymnasium
import gymnasium as classic_gym  # TD-MPC2's Timeout/TensorWrapper also subclass gymnasium.Wrapper
import numpy as np

from envs.wrappers.timeout import Timeout
from envs.wrappers.action_repeat import ActionRepeat

# Make sure the Metaworld repo root is on sys.path so that we can import
# `metaworld` and `metaworld.wrappers` regardless of the working directory.
_BASE_DIR = Path(__file__).resolve().parents[4]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import metaworld  # noqa: F401  # registers the "Meta-World/MT1" entry
from metaworld.wrappers import ProprioMultiImageObsWrapper



DEFAULT_CAMERAS = ("topview", "front", "gripperPOV")
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MAX_EPISODE_STEPS = 250
DEFAULT_ACTION_REPEAT = 2


class RewardTuningWrapperV2(gymnasium.Wrapper):
    """Scale rewards from [-1, 1] into [-1, 0] to match DreamerV3 setup."""

    def __init__(
        self,
        env: gymnasium.Env,
        original_reward_range: tuple = (-1.0, 1.0),
        target_reward_range: tuple = (-1.0, 0.0),
    ):
        super().__init__(env)
        self.orig_min, self.orig_max = original_reward_range
        self.target_min, self.target_max = target_reward_range

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        normed = (reward - self.orig_min) / (self.orig_max - self.orig_min)
        scaled = self.target_min + normed * (self.target_max - self.target_min)
        return obs, scaled, terminated, truncated, info


class _Gymnasium5To4(classic_gym.Wrapper):
    """Convert Gymnasium's (obs, reward, terminated, truncated, info) to 4-tuple."""

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info


class _DictObsAdapter(classic_gym.Wrapper):
    """Rename Metaworld-wrapper dict keys and convert image to CHW for TD-MPC2.

    Input dict obs has keys:
      - 'proprio'     (7,)      float32
      - 'image'       (H,W,3N)  uint8
      - 'original_obs' ...      (dropped — TD-MPC2 doesn't need it)

    Output dict obs has keys:
      - 'state' (7,)       float32     proprio
      - 'rgb'   (3N,H,W)   uint8       channels-first for TD-MPC2 conv
    """

    def __init__(self, env: classic_gym.Env):
        super().__init__(env)
        orig = env.observation_space.spaces
        proprio_space = orig["proprio"]
        img_hw_c = orig["image"]
        h, w, c = img_hw_c.shape
        self.observation_space = classic_gym.spaces.Dict(
            {
                "state": classic_gym.spaces.Box(
                    low=proprio_space.low,
                    high=proprio_space.high,
                    shape=proprio_space.shape,
                    dtype=np.float32,
                ),
                "rgb": classic_gym.spaces.Box(
                    low=0, high=255, shape=(c, h, w), dtype=np.uint8
                ),
            }
        )

    def _convert(self, obs):
        img = obs["image"]
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        return {"state": obs["proprio"].astype(np.float32), "rgb": img}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._convert(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._convert(obs), reward, done, info

    def render(self, *args, **kwargs):
        # Delegate to the underlying gymnasium wrapper's render.
        return self.env.render(*args, **kwargs)


class _AnnotateTerminated(gymnasium.Wrapper):
    """Preserve the distinction between terminated and truncated in `info`
    before the subsequent _Gymnasium5To4 wrapper collapses them into `done`."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return obs, reward, terminated, truncated, info


class _MetaworldInfoShim(classic_gym.Wrapper):
    """Ensure `info` contains the keys TD-MPC2's TensorWrapper expects."""

    def __init__(self, env: classic_gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # TD-MPC2's TensorWrapper accesses info['success'] and info['terminated'].
        info.setdefault("success", float(info.get("success", 0.0)))
        info.setdefault("terminated", False)
        return obs, reward, done, info


_TASK_PREFIXES = ("metaworld_", "mw-")


def _is_metaworld_task(task: str) -> bool:
    return isinstance(task, str) and any(task.startswith(p) for p in _TASK_PREFIXES)


def _strip_task_prefix(task: str) -> str:
    for p in _TASK_PREFIXES:
        if task.startswith(p):
            return task[len(p):]
    raise ValueError(
        f"Expected task to start with one of {_TASK_PREFIXES}, got: {task}"
    )


def make_env(cfg):
    """Make the DreamerV3-aligned Metaworld environment for TD-MPC2."""
    task = cfg.task
    if not _is_metaworld_task(task):
        raise ValueError("Unknown task:", task)
    env_name = _strip_task_prefix(task)

    cameras = list(cfg.get("cameras", DEFAULT_CAMERAS) or DEFAULT_CAMERAS)
    image_size = int(cfg.get("image_size", DEFAULT_IMAGE_SIZE) or DEFAULT_IMAGE_SIZE)
    max_episode_steps = int(
        cfg.get("max_episode_steps", DEFAULT_MAX_EPISODE_STEPS) or DEFAULT_MAX_EPISODE_STEPS
    )
    action_repeat = int(
        cfg.get("action_repeat", DEFAULT_ACTION_REPEAT) or DEFAULT_ACTION_REPEAT
    )

    # The inner gymnasium-level max steps must account for the outer
    # action-repeat factor so that timeouts fire at the intended env-step count.
    inner_max_steps = max_episode_steps * action_repeat

    env = gymnasium.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode="rgb_array",
        max_episode_steps=inner_max_steps,
    )
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=image_size,
        image_width=image_size,
        camera_names=cameras,
    )
    env = RewardTuningWrapperV2(
        env,
        original_reward_range=(-1.0, 1.0),
        target_reward_range=(-1.0, 0.0),
    )
    env = _AnnotateTerminated(env)
    env = _Gymnasium5To4(env)
    env = _DictObsAdapter(env)
    env = _MetaworldInfoShim(env)
    if action_repeat > 1:
        env = ActionRepeat(env, repeat=action_repeat)
    env = Timeout(env, max_episode_steps=max_episode_steps)
    return env
