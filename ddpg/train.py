import argparse
import os
import json
import copy
import torch
import torch.multiprocessing as mp
from multiprocessing import Value

from common.misc_util import boolean_flag, str2params, create_if_need
from common.env_wrappers import create_env
from common.torch_util import activations, hard_update

from ddpg.model import create_model, create_act_update_fns, train_multi_thread, \
    train_single_thread, play_single_thread


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--difficulty', type=int, default=2)
    parser.add_argument('--max-obstacles', type=int, default=3)

    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--num-train-threads', type=int, default=1)

    boolean_flag(parser, "ddpg-wrapper", default=False)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--fail-reward', type=float, default=0.0)
    parser.add_argument('--reward-scale', type=float, default=1.)
    boolean_flag(parser, "flip-state-action", default=False)

    for agent in ["actor", "critic"]:
        parser.add_argument('--{}-layers'.format(agent), type=str, default="64-64")
        parser.add_argument('--{}-activation'.format(agent), type=str, default="relu")
        boolean_flag(parser, "{}-layer-norm".format(agent), default=False)
        boolean_flag(parser, "{}-parameters-noise".format(agent), default=False)
        boolean_flag(parser, "{}-parameters-noise-factorised".format(agent), default=False)

        parser.add_argument('--{}-lr'.format(agent), type=float, default=1e-3)
        parser.add_argument('--{}-lr-end'.format(agent), type=float, default=5e-5)

        parser.add_argument('--restore-{}-from'.format(agent), type=str, default=None)

    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--loss-type', type=str, default="quadric-linear")
    parser.add_argument('--grad-clip', type=float, default=10.)

    parser.add_argument('--tau', default=0.01, type=float)

    parser.add_argument('--train-steps', type=int, default=int(1e4))
    parser.add_argument('--batch-size', type=int, default=256)  # per worker

    parser.add_argument('--buffer-size', type=int, default=int(1e6))

    boolean_flag(parser, "prioritized-replay", default=False)
    parser.add_argument('--prioritized-replay-alpha', default=0.6, type=float)
    parser.add_argument('--prioritized-replay-beta0', default=0.4, type=float)

    parser.add_argument('--initial-epsilon', default=1., type=float)
    parser.add_argument('--final-epsilon', default=0.01, type=float)
    parser.add_argument('--max-episodes', default=int(1e4), type=int)
    parser.add_argument('--max-update-steps', default=int(5e6), type=int)
    parser.add_argument('--epsilon-cycle-len', default=int(2e2), type=int)

    parser.add_argument('--max-train-days', default=int(1e1), type=int)

    parser.add_argument('--rp-type', default="ornstein-uhlenbeck", type=str)
    parser.add_argument('--rp-theta', default=0.15, type=float)
    parser.add_argument('--rp-sigma', default=0.2, type=float)
    parser.add_argument('--rp-sigma-min', default=0.15, type=float)
    parser.add_argument('--rp-mu', default=0.0, type=float)

    parser.add_argument('--clip-delta', type=int, default=10)
    parser.add_argument('--save-step', type=int, default=int(1e4))

    parser.add_argument('--restore-args-from', type=str, default=None)

    return parser.parse_args()


def restore_args(args):
    with open(args.restore_args_from, "r") as fin:
        params = json.load(fin)

    del params["seed"]
    del params["difficulty"]
    del params["max_obstacles"]

    del params["logdir"]
    del params["num_threads"]
    del params["num_train_threads"]

    del params["skip_frames"]

    for agent in ["actor", "critic"]:
        del params["{}_lr".format(agent)]
        del params["{}_lr_end".format(agent)]
        del params["restore_{}_from".format(agent)]

    del params["grad_clip"]

    del params["tau"]

    del params["train_steps"]
    del params["batch_size"]

    del params["buffer_size"]

    del params["prioritized_replay"]
    del params["prioritized_replay_alpha"]
    del params["prioritized_replay_beta0"]

    del params["initial_epsilon"]
    del params["final_epsilon"]
    del params["max_episodes"]
    del params["max_update_steps"]
    del params["epsilon_cycle_len"]

    del params["max_train_days"]

    del params["rp_type"]
    del params["rp_theta"]
    del params["rp_sigma"]
    del params["rp_sigma_min"]
    del params["rp_mu"]

    del params["clip_delta"]
    del params["save_step"]

    del params["restore_args_from"]

    for key, value in params.items():
        setattr(args, key, value)
    return args


def train(args, model_fn, act_update_fns, multi_thread, train_single, play_single):
    create_if_need(args.logdir)

    if args.restore_args_from is not None:
        args = restore_args(args)

    with open("{}/args.json".format(args.logdir), "w") as fout:
        json.dump(vars(args), fout, indent=4, ensure_ascii=False, sort_keys=True)

    env = create_env(args)

    if args.flip_state_action and hasattr(env, "state_transform"):
        args.flip_states = env.state_transform.flip_states
        args.batch_size = args.batch_size // 2

    args.n_action = env.action_space.shape[0]
    args.n_observation = env.observation_space.shape[0]

    args.actor_layers = str2params(args.actor_layers)
    args.critic_layers = str2params(args.critic_layers)

    args.actor_activation = activations[args.actor_activation]
    args.critic_activation = activations[args.critic_activation]

    actor, critic = model_fn(args)

    if args.restore_actor_from is not None:
        actor.load_state_dict(torch.load(args.restore_actor_from))
    if args.restore_critic_from is not None:
        critic.load_state_dict(torch.load(args.restore_critic_from))

    actor.train()
    critic.train()
    actor.share_memory()
    critic.share_memory()

    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    hard_update(target_actor, actor)
    hard_update(target_critic, critic)

    target_actor.train()
    target_critic.train()
    target_actor.share_memory()
    target_critic.share_memory()

    _, _, save_fn = act_update_fns(actor, critic, target_actor, target_critic, args)

    processes = []
    best_reward = Value("f", 0.0)
    try:
        if args.num_threads == args.num_train_threads:
            for rank in range(args.num_threads):
                args.thread = rank
                p = mp.Process(
                    target=multi_thread,
                    args=(actor, critic, target_actor, target_critic, args, act_update_fns,
                          best_reward))
                p.start()
                processes.append(p)
        else:
            global_episode = Value("i", 0)
            global_update_step = Value("i", 0)
            episodes_queue = mp.Queue()
            for rank in range(args.num_threads):
                args.thread = rank
                if rank < args.num_train_threads:
                    p = mp.Process(
                        target=train_single,
                        args=(actor, critic, target_actor, target_critic, args, act_update_fns,
                              global_episode, global_update_step, episodes_queue))
                else:
                    p = mp.Process(
                        target=play_single,
                        args=(actor, critic, target_actor, target_critic, args, act_update_fns,
                              global_episode, global_update_step, episodes_queue,
                              best_reward))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        pass

    save_fn()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    args = parse_args()
    train(args,
          create_model,
          create_act_update_fns,
          train_multi_thread,
          train_single_thread,
          play_single_thread)
