import numpy as np
import gym
from gym.spaces import Box
from osim.env import RunEnv

from common.state_transform import StateVelCentr


class DdpgWrapper(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env)
        self.state_transform = StateVelCentr(
            obstacles_mode='standard',
            exclude_centr=True,
            vel_states=[])
        self.observation_space = Box(-1000, 1000, self.state_transform.state_size)
        self.skip_frames = args.skip_frames
        self.reward_scale = args.reward_scale
        self.fail_reward = args.fail_reward
        # [-1, 1] <-> [0, 1]
        action_mean = .5
        action_std = .5
        self.normalize_action = lambda x: (x - action_mean) / action_std
        self.denormalize_action = lambda x: x * action_std + action_mean

    def reset(self, **kwargs):
        return self._reset(**kwargs)

    def _reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.env_step = 0
        self.state_transform.reset()
        observation, _ = self.state_transform.process(observation)
        observation = self.observation(observation)
        return observation

    def _step(self, action):
        action = self.denormalize_action(action)
        total_reward = 0.
        for _ in range(self.skip_frames):
            observation, reward, done, _ = self.env.step(action)
            observation, obst_rew = self.state_transform.process(observation)
            total_reward += reward + obst_rew
            self.env_step += 1
            if done:
                if self.env_step < 1000:  # hardcoded
                    total_reward += self.fail_reward
                break

        observation = self.observation(observation)
        total_reward *= self.reward_scale
        return observation, total_reward, done, None

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        observation = np.array(observation, dtype=np.float32)
        return observation


def create_env(args):
    env = RunEnv(visualize=False, max_obstacles=args.max_obstacles)

    if hasattr(args, "baseline_wrapper") or hasattr(args, "ddpg_wrapper"):
        env = DdpgWrapper(env, args)

    return env


def create_observation_handler(args):

    if hasattr(args, "baseline_wrapper") or hasattr(args, "ddpg_wrapper"):
        state_transform = StateVelCentr(
            obstacles_mode='standard',
            exclude_centr=True,
            vel_states=[])

        def observation_handler(observation, previous_action=None):
            observation = np.array(observation, dtype=np.float32)
            observation, _ = state_transform.process(observation)
            return observation
    else:
        def observation_handler(observation, previous_action=None):
            observation = np.array(observation, dtype=np.float32)
            return observation

    return observation_handler


def create_action_handler(args):
    action_mean = .5
    action_std = .5
    action_handler = lambda x: x * action_std + action_mean
    return action_handler
