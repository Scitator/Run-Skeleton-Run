import os
import torch
import copy
from multiprocessing import Value

from common.misc_util import str2params, create_if_need
from common.env_wrappers import create_env
from common.torch_util import activations, hard_update

from ddpg.model import create_model, create_act_update_fns, train_multi_thread
from ddpg.train import parse_args


def debug(args, model_fn, act_update_fns, multi_thread):
    create_if_need(args.logdir)
    env = create_env(args)

    if args.flip_state_action and hasattr(env, "state_transform"):
        args.flip_states = env.state_transform.flip_states

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
    critic.train()
    target_actor.share_memory()
    target_critic.share_memory()

    _, _, save_fn = act_update_fns(actor, critic, target_actor, target_critic, args)

    args.thread = 0
    best_reward = Value("f", 0.0)
    multi_thread(actor, critic, target_actor, target_critic, args, act_update_fns, best_reward)

    save_fn()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    args = parse_args()
    debug(
        args,
        create_model,
        create_act_update_fns,
        train_multi_thread)
