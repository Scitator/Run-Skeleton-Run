import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pprint import pprint

from osim.env import RunEnv
from osim.http.client import Client

from common.misc_util import boolean_flag, query_yes_no
from common.env_wrappers import create_observation_handler, create_action_handler, create_env

from ddpg.train import str2params, activations
from ddpg.model import create_model, create_act_update_fns


REMOTE_BASE = 'http://grader.crowdai.org:1729'
ACTION_SHAPE = 18
SEEDS = [
    3834825972, 3049289152, 3538742899, 2904257823, 4011088434,
    2684066875, 781202090, 1691535473, 898088606, 1301477286
]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--restore-args-from', type=str, default=None)
    parser.add_argument('--restore-actor-from', type=str, default=None)
    parser.add_argument('--restore-critic-from', type=str, default=None)

    parser.add_argument('--max-obstacles', type=int, default=3)
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--token', type=str, default=None)

    boolean_flag(parser, "visualize", default=False)
    boolean_flag(parser, "submit", default=False)

    return parser.parse_args()


def restore_args(args):
    with open(args.restore_args_from, "r") as fin:
        params = json.load(fin)

    unwanted = [
        "max_obstacles",
        "restore_args_from",
        "restore_actor_from",
        "restore_critic_from"
    ]

    for unwanted_key in unwanted:
        value = params.pop(unwanted_key, None)
        if value is not None:
            del value

    for key, value in params.items():
        setattr(args, key, value)
    return args


def submit(actor, critic, args, act_update_fn):
    act_fn, _, _ = act_update_fn(actor, critic, None, None, args)

    client = Client(REMOTE_BASE)

    all_episode_metrics = []

    episode_metrics = {
        "reward": 0.0,
        "step": 0,
    }

    observation_handler = create_observation_handler(args)
    action_handler = create_action_handler(args)
    observation = client.env_create(args.token)
    action = np.zeros(ACTION_SHAPE, dtype=np.float32)
    observation = observation_handler(observation, action)

    submitted = False
    while not submitted:
        print(episode_metrics["reward"])
        action = act_fn(observation)

        observation, reward, done, _ = client.env_step(action_handler(action).tolist())

        episode_metrics["reward"] += reward
        episode_metrics["step"] += 1

        if done:
            all_episode_metrics.append(episode_metrics)

            episode_metrics = {
                "reward": 0.0,
                "step": 0,
            }

            observation_handler = create_observation_handler(args)
            action_handler = create_action_handler(args)
            observation = client.env_create(args.token)

            if not observation:
                submitted = True
                break

            action = np.zeros(ACTION_SHAPE, dtype=np.float32)
            observation = observation_handler(observation, action)
        else:
            observation = observation_handler(observation, action)

    df = pd.DataFrame(all_episode_metrics)
    pprint(df.describe())

    if query_yes_no("Submit?"):
        client.submit()


def test(actor, critic, args, act_update_fn):
    act_fn, _, _ = act_update_fn(actor, critic, None, None, args)
    env = RunEnv(visualize=args.visualize, max_obstacles=args.max_obstacles)

    all_episode_metrics = []
    for episode in range(args.num_episodes):
        episode_metrics = {
            "reward": 0.0,
            "step": 0,
        }

        observation_handler = create_observation_handler(args)
        action_handler = create_action_handler(args)
        observation = env.reset(difficulty=2, seed=SEEDS[episode % len(SEEDS)])
        action = np.zeros(ACTION_SHAPE, dtype=np.float32)
        observation = observation_handler(observation, action)

        done = False
        while not done:
            print(episode_metrics["reward"])
            action = act_fn(observation)

            observation, reward, done, _ = env.step(action_handler(action))

            episode_metrics["reward"] += reward
            episode_metrics["step"] += 1

            if done:
                break

            observation = observation_handler(observation, action)

        all_episode_metrics.append(episode_metrics)

    df = pd.DataFrame(all_episode_metrics)
    pprint(df.describe())


def submit_or_test(args, model_fn, act_update_fn, submit_fn, test_fn):
    args = restore_args(args)
    env = create_env(args)

    args.n_action = env.action_space.shape[0]
    args.n_observation = env.observation_space.shape[0]

    args.actor_layers = str2params(args.actor_layers)
    args.critic_layers = str2params(args.critic_layers)

    args.actor_activation = activations[args.actor_activation]
    args.critic_activation = activations[args.critic_activation]

    actor, critic = model_fn(args)
    actor.load_state_dict(torch.load(args.restore_actor_from))
    critic.load_state_dict(torch.load(args.restore_critic_from))

    if args.submit:
        submit_fn(actor, critic, args, act_update_fn)
    else:
        test_fn(actor, critic, args, act_update_fn)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    args = parse_args()
    submit_or_test(args, create_model, create_act_update_fns, submit, test)
