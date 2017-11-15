#!/usr/bin/env python
# noinspection PyUnresolvedReferences

import os
import json
import argparse
from mpi4py import MPI

from common.misc_util import boolean_flag, str2params, create_if_need
from common.misc_util import set_global_seeds
from common.env_wrappers import create_env

from baselines.nets import Actor
from baselines import trpo, ppo


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--agent',
        type=str,
        default="trpo",
        choices=["trpo", "ppo"],
        help='Which agent to use. (default: %(default)s)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--difficulty', type=int, default=2)
    parser.add_argument('--max-obstacles', type=int, default=3)

    parser.add_argument('--logdir', type=str, default="./logs")

    boolean_flag(parser, "baseline-wrapper", default=False)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--fail-reward', type=float, default=0.0)

    parser.add_argument('--hid-size', type=int, default=64)
    parser.add_argument('--num-hid-layers', type=int, default=2)

    parser.add_argument('--gamma', type=float, default=0.96)

    parser.add_argument('--restore-args-from', type=str, default=None)
    parser.add_argument('--restore-actor-from', type=str, default=None)

    parser.add_argument(
        '--max-train-days',
        default=int(1e1),
        type=int)

    args = parser.parse_args()
    return args


def restore_params(args):
    with open(args.restore_args_from, "r") as fin:
        params = json.load(fin)

    del params["seed"]
    del params["difficulty"]
    del params["max_obstacles"]

    del params["skip_frames"]

    del params["restore_args_from"]
    del params["restore_actor_from"]

    for key, value in params.items():
        setattr(args, key, value)
    return args
        

def train(args):
    import baselines.baselines_common.tf_util as U

    sess = U.single_threaded_session()
    sess.__enter__()

    if args.restore_args_from is not None:
        args = restore_params(args)

    rank = MPI.COMM_WORLD.Get_rank()

    workerseed = args.seed + 241 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    def policy_fn(name, ob_space, ac_space):
        return Actor(
            name=name,
            ob_space=ob_space, ac_space=ac_space,
            hid_size=args.hid_size, num_hid_layers=args.num_hid_layers,
            noise_type=args.noise_type)

    env = create_env(args)
    env.seed(workerseed)

    if rank == 0:
        create_if_need(args.logdir)
        with open("{}/args.json".format(args.logdir), "w") as fout:
            json.dump(vars(args), fout, indent=4, ensure_ascii=False, sort_keys=True)

    try:
        args.thread = rank
        if args.agent == "trpo":
            trpo.learn(
                env, policy_fn, args,
                timesteps_per_batch=1024,
                gamma=args.gamma,
                lam=0.98,
                max_kl=0.01,
                cg_iters=10,
                cg_damping=0.1,
                vf_iters=5,
                vf_stepsize=1e-3)
        elif args.agent == "ppo":
            # optimal settings:
            # timesteps_per_batch = optim_epochs *  optim_batchsize
            ppo.learn(
                env, policy_fn, args,
                timesteps_per_batch=256,
                gamma=args.gamma,
                lam=0.95,
                clip_param=0.2,
                entcoeff=0.0,
                optim_epochs=4,
                optim_stepsize=3e-4,
                optim_batchsize=64,
                schedule='constant')
        else:
            raise NotImplementedError
    except KeyboardInterrupt:
        print("closing envs...")

    env.close()


if __name__ == '__main__':
    args = parse_args()
    args.noise_type = "gaussian"
    train(args)
