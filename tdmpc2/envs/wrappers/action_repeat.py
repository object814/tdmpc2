import gym


class ActionRepeat(gym.Wrapper):
    """Repeat the same action `repeat` times and sum rewards.

    Mirrors the 2-step accumulation in TD-MPC2's original MetaWorldWrapper
    (third_party/tdmpc2/tdmpc2/envs/metaworld.py), but works over the
    classic-gym API (obs, reward, done, info).
    """

    def __init__(self, env, repeat: int = 2):
        super().__init__(env)
        assert repeat >= 1, "action_repeat must be >= 1"
        self._repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        obs, done, info = None, False, {}
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if done:
                break
        return obs, total_reward, done, info
