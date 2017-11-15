import random
import numpy as np
import torch
import queue as py_queue
import time
import torch.nn as nn
from pprint import pprint

from ddpg.nets import Actor, Critic
from common.torch_util import to_numpy, to_tensor, soft_update
from common.misc_util import create_if_need, set_global_seeds
from common.logger import Logger
from common.buffers import create_buffer
from common.loss import create_loss, create_decay_fn
from common.env_wrappers import create_env
from common.random_process import create_random_process


def create_model(args):
    actor = Actor(
        args.n_observation, args.n_action, args.actor_layers,
        activation=args.actor_activation,
        layer_norm=args.actor_layer_norm,
        parameters_noise=args.actor_parameters_noise,
        parameters_noise_factorised=args.actor_parameters_noise_factorised,
        last_activation=nn.Tanh)
    critic = Critic(
        args.n_observation, args.n_action, args.critic_layers,
        activation=args.critic_activation,
        layer_norm=args.critic_layer_norm,
        parameters_noise=args.critic_parameters_noise,
        parameters_noise_factorised=args.critic_parameters_noise_factorised)

    pprint(actor)
    pprint(critic)

    return actor, critic


def create_act_update_fns(actor, critic, target_actor, target_critic, args):
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    criterion = create_loss(args)

    low_action_boundary = -1.
    high_action_boundary = 1.

    def act_fn(observation, noise=0):
        nonlocal actor
        action = to_numpy(actor(to_tensor(np.array([observation], dtype=np.float32)))).squeeze(0)
        action += noise
        action = np.clip(action, low_action_boundary, high_action_boundary)
        return action

    def update_fn(
            observations, actions, rewards, next_observations, dones, weights,
            actor_lr=1e-4, critic_lr=1e-3):
        nonlocal actor, critic, target_actor, target_critic, actor_optim, critic_optim

        if hasattr(args, "flip_states"):
            observations_flip = args.flip_states(observations)
            next_observations_flip = args.flip_states(next_observations)
            actions_flip = np.zeros_like(actions)
            actions_flip[:, :args.n_action // 2] = actions[:, args.n_action // 2:]
            actions_flip[:, args.n_action // 2:] = actions[:, :args.n_action // 2]

            observations = np.concatenate((observations, observations_flip))
            actions = np.concatenate((actions, actions_flip))
            rewards = np.tile(rewards.ravel(), 2)
            next_observations = np.concatenate((next_observations, next_observations_flip))
            dones = np.tile(dones.ravel(), 2)

        dones = dones[:, None].astype(np.bool)
        rewards = rewards[:, None].astype(np.float32)

        dones = to_tensor(np.invert(dones).astype(np.float32))
        rewards = to_tensor(rewards)
        weights = to_tensor(weights, requires_grad=False)

        next_v_values = target_critic(
            to_tensor(next_observations, volatile=True),
            target_actor(to_tensor(next_observations, volatile=True)),
        )
        next_v_values.volatile = False

        reward_predicted = dones * args.gamma * next_v_values
        td_target = rewards + reward_predicted

        # Critic update
        critic.zero_grad()

        v_values = critic(to_tensor(observations), to_tensor(actions))
        value_loss = criterion(v_values, td_target, weights=weights)
        value_loss.backward()

        torch.nn.utils.clip_grad_norm(critic.parameters(), args.grad_clip)
        for param_group in critic_optim.param_groups:
            param_group["lr"] = critic_lr

        critic_optim.step()

        # Actor update
        actor.zero_grad()

        policy_loss = -critic(
            to_tensor(observations),
            actor(to_tensor(observations))
        )

        policy_loss = torch.mean(policy_loss * weights)
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm(actor.parameters(), args.grad_clip)
        for param_group in actor_optim.param_groups:
            param_group["lr"] = actor_lr

        actor_optim.step()

        # Target update
        soft_update(target_actor, actor, args.tau)
        soft_update(target_critic, critic, args.tau)

        metrics = {
            "value_loss": value_loss,
            "policy_loss": policy_loss
        }

        td_v_values = critic(
            to_tensor(observations, volatile=True, requires_grad=False),
            to_tensor(actions, volatile=True, requires_grad=False))
        td_error = td_target - td_v_values

        info = {
            "td_error": to_numpy(td_error)
        }

        return metrics, info

    def save_fn(episode=None):
        nonlocal actor, critic
        if episode is None:
            save_path = args.logdir
        else:
            save_path = "{}/episode_{}".format(args.logdir, episode)
            create_if_need(save_path)
        torch.save(actor.state_dict(), "{}/actor_state_dict.pkl".format(save_path))
        torch.save(critic.state_dict(), "{}/critic_state_dict.pkl".format(save_path))
        torch.save(target_actor.state_dict(), "{}/target_actor_state_dict.pkl".format(save_path))
        torch.save(target_critic.state_dict(), "{}/target_critic_state_dict.pkl".format(save_path))

    return act_fn, update_fn, save_fn


def train_multi_thread(actor, critic, target_actor, target_critic, args, prepare_fn, best_reward):
    workerseed = args.seed + 241 * args.thread
    set_global_seeds(workerseed)

    args.logdir = "{}/thread_{}".format(args.logdir, args.thread)
    create_if_need(args.logdir)

    act_fn, update_fn, save_fn = prepare_fn(actor, critic, target_actor, target_critic, args)
    logger = Logger(args.logdir)

    buffer = create_buffer(args)
    if args.prioritized_replay:
        beta_deacy_fn = create_decay_fn(
            "linear",
            initial_value=args.prioritized_replay_beta0,
            final_value=1.0,
            max_step=args.max_episodes)

    env = create_env(args)
    random_process = create_random_process(args)

    actor_learning_rate_decay_fn = create_decay_fn(
        "linear",
        initial_value=args.actor_lr,
        final_value=args.actor_lr_end,
        max_step=args.max_episodes)
    critic_learning_rate_decay_fn = create_decay_fn(
        "linear",
        initial_value=args.critic_lr,
        final_value=args.critic_lr_end,
        max_step=args.max_episodes)

    epsilon_cycle_len = random.randint(args.epsilon_cycle_len // 2, args.epsilon_cycle_len * 2)

    epsilon_decay_fn = create_decay_fn(
        "cycle",
        initial_value=args.initial_epsilon,
        final_value=args.final_epsilon,
        cycle_len=epsilon_cycle_len,
        num_cycles=args.max_episodes // epsilon_cycle_len)

    episode = 0
    step = 0
    start_time = time.time()
    while episode < args.max_episodes:
        if episode % 100 == 0:
            env = create_env(args)
        seed = random.randrange(2 ** 32 - 2)

        actor_lr = actor_learning_rate_decay_fn(episode)
        critic_lr = critic_learning_rate_decay_fn(episode)
        epsilon = min(args.initial_epsilon, max(args.final_epsilon, epsilon_decay_fn(episode)))

        episode_metrics = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "reward": 0.0,
            "step": 0,
            "epsilon": epsilon
        }

        observation = env.reset(seed=seed, difficulty=args.difficulty)
        random_process.reset_states()
        done = False

        while not done:
            action = act_fn(observation, noise=epsilon*random_process.sample())
            next_observation, reward, done, _ = env.step(action)

            buffer.add(observation, action, reward, next_observation, done)
            episode_metrics["reward"] += reward
            episode_metrics["step"] += 1

            if len(buffer) >= args.train_steps:

                if args.prioritized_replay:
                    (tr_observations, tr_actions, tr_rewards, tr_next_observations, tr_dones,
                     weights, batch_idxes) = \
                        buffer.sample(batch_size=args.batch_size, beta=beta_deacy_fn(episode))
                else:
                    (tr_observations, tr_actions, tr_rewards, tr_next_observations, tr_dones) = \
                        buffer.sample(batch_size=args.batch_size)
                    weights, batch_idxes = np.ones_like(tr_rewards), None

                step_metrics, step_info = update_fn(
                    tr_observations, tr_actions, tr_rewards,
                    tr_next_observations, tr_dones,
                    weights, actor_lr, critic_lr)

                if args.prioritized_replay:
                    new_priorities = np.abs(step_info["td_error"]) + 1e-6
                    buffer.update_priorities(batch_idxes, new_priorities)

                for key, value in step_metrics.items():
                    value = to_numpy(value)[0]
                    episode_metrics[key] += value

            observation = next_observation

        episode += 1

        if episode_metrics["reward"] > 15.0 * args.reward_scale \
                and episode_metrics["reward"] > best_reward.value:
            best_reward.value = episode_metrics["reward"]
            logger.scalar_summary("best reward", best_reward.value, episode)
            save_fn(episode)

        step += episode_metrics["step"]
        elapsed_time = time.time() - start_time

        for key, value in episode_metrics.items():
            value = value if "loss" not in key else value / episode_metrics["step"]
            logger.scalar_summary(key, value, episode)
        logger.scalar_summary(
            "episode per minute",
            episode / elapsed_time * 60,
            episode)
        logger.scalar_summary(
            "step per second",
            step / elapsed_time,
            episode)
        logger.scalar_summary("actor lr", actor_lr, episode)
        logger.scalar_summary("critic lr", critic_lr, episode)

        if episode % args.save_step == 0:
            save_fn(episode)

        if elapsed_time > 86400 * args.max_train_days:
            episode = args.max_episodes + 1

    save_fn(episode)

    raise KeyboardInterrupt


def train_single_thread(
        actor, critic, target_actor, target_critic, args, prepare_fn,
        global_episode, global_update_step, episodes_queue):
    workerseed = args.seed + 241 * args.thread
    set_global_seeds(workerseed)

    args.logdir = "{}/thread_{}".format(args.logdir, args.thread)
    create_if_need(args.logdir)

    _, update_fn, save_fn = prepare_fn(actor, critic, target_actor, target_critic, args)

    logger = Logger(args.logdir)

    buffer = create_buffer(args)

    if args.prioritized_replay:
        beta_deacy_fn = create_decay_fn(
            "linear",
            initial_value=args.prioritized_replay_beta0,
            final_value=1.0,
            max_step=args.max_update_steps)

    actor_learning_rate_decay_fn = create_decay_fn(
        "linear",
        initial_value=args.actor_lr,
        final_value=args.actor_lr_end,
        max_step=args.max_update_steps)
    critic_learning_rate_decay_fn = create_decay_fn(
        "linear",
        initial_value=args.critic_lr,
        final_value=args.critic_lr_end,
        max_step=args.max_update_steps)

    update_step = 0
    received_examples = 1  # just hack
    while global_episode.value < args.max_episodes * (args.num_threads - args.num_train_threads) \
            and global_update_step.value < args.max_update_steps * args.num_train_threads:
        actor_lr = actor_learning_rate_decay_fn(update_step)
        critic_lr = critic_learning_rate_decay_fn(update_step)

        actor_lr = min(args.actor_lr, max(args.actor_lr_end, actor_lr))
        critic_lr = min(args.critic_lr, max(args.critic_lr_end, critic_lr))

        while True:
            try:
                replay = episodes_queue.get_nowait()
                for (observation, action, reward, next_observation, done) in replay:
                    buffer.add(observation, action, reward, next_observation, done)
                received_examples += len(replay)
            except py_queue.Empty:
                break

        if len(buffer) >= args.train_steps:
            if args.prioritized_replay:
                beta = beta_deacy_fn(update_step)
                beta = min(1.0, max(args.prioritized_replay_beta0, beta))
                (tr_observations, tr_actions, tr_rewards, tr_next_observations, tr_dones,
                 weights, batch_idxes) = \
                    buffer.sample(
                        batch_size=args.batch_size,
                        beta=beta)
            else:
                (tr_observations, tr_actions, tr_rewards, tr_next_observations, tr_dones) = \
                    buffer.sample(batch_size=args.batch_size)
                weights, batch_idxes = np.ones_like(tr_rewards), None

            step_metrics, step_info = update_fn(
                tr_observations, tr_actions, tr_rewards,
                tr_next_observations, tr_dones,
                weights, actor_lr, critic_lr)

            update_step += 1
            global_update_step.value += 1

            if args.prioritized_replay:
                new_priorities = np.abs(step_info["td_error"]) + 1e-6
                buffer.update_priorities(batch_idxes, new_priorities)

            for key, value in step_metrics.items():
                value = to_numpy(value)[0]
                logger.scalar_summary(key, value, update_step)

            logger.scalar_summary("actor lr", actor_lr, update_step)
            logger.scalar_summary("critic lr", critic_lr, update_step)

            if update_step % args.save_step == 0:
                save_fn(update_step)
        else:
            time.sleep(1)

        logger.scalar_summary("buffer size", len(buffer), global_episode.value)
        logger.scalar_summary(
            "updates per example",
            update_step * args.batch_size / received_examples,
            global_episode.value)

    save_fn(update_step)

    raise KeyboardInterrupt


def play_single_thread(
        actor, critic, target_actor, target_critic, args, prepare_fn,
        global_episode, global_update_step, episodes_queue,
        best_reward):
    workerseed = args.seed + 241 * args.thread
    set_global_seeds(workerseed)

    args.logdir = "{}/thread_{}".format(args.logdir, args.thread)
    create_if_need(args.logdir)

    act_fn, _, save_fn = prepare_fn(actor, critic, target_actor, target_critic, args)

    logger = Logger(args.logdir)
    env = create_env(args)
    random_process = create_random_process(args)

    epsilon_cycle_len = random.randint(args.epsilon_cycle_len // 2, args.epsilon_cycle_len * 2)

    epsilon_decay_fn = create_decay_fn(
        "cycle",
        initial_value=args.initial_epsilon,
        final_value=args.final_epsilon,
        cycle_len=epsilon_cycle_len,
        num_cycles=args.max_episodes // epsilon_cycle_len)

    episode = 1
    step = 0
    start_time = time.time()
    while global_episode.value < args.max_episodes * (args.num_threads - args.num_train_threads) \
            and global_update_step.value < args.max_update_steps * args.num_train_threads:
        if episode % 100 == 0:
            env = create_env(args)
        seed = random.randrange(2 ** 32 - 2)

        epsilon = min(args.initial_epsilon, max(args.final_epsilon, epsilon_decay_fn(episode)))

        episode_metrics = {
            "reward": 0.0,
            "step": 0,
            "epsilon": epsilon
        }

        observation = env.reset(seed=seed, difficulty=args.difficulty)
        random_process.reset_states()
        done = False

        replay = []
        while not done:
            action = act_fn(observation, noise=epsilon * random_process.sample())
            next_observation, reward, done, _ = env.step(action)

            replay.append((observation, action, reward, next_observation, done))
            episode_metrics["reward"] += reward
            episode_metrics["step"] += 1

            observation = next_observation

        episodes_queue.put(replay)

        episode += 1
        global_episode.value += 1

        if episode_metrics["reward"] > best_reward.value:
            best_reward.value = episode_metrics["reward"]
            logger.scalar_summary("best reward", best_reward.value, episode)

            if episode_metrics["reward"] > 15.0 * args.reward_scale:
                save_fn(episode)

        step += episode_metrics["step"]
        elapsed_time = time.time() - start_time

        for key, value in episode_metrics.items():
            logger.scalar_summary(key, value, episode)
        logger.scalar_summary(
            "episode per minute",
            episode / elapsed_time * 60,
            episode)
        logger.scalar_summary(
            "step per second",
            step / elapsed_time,
            episode)

        if elapsed_time > 86400 * args.max_train_days:
            global_episode.value = args.max_episodes * (args.num_threads - args.num_train_threads) + 1

    raise KeyboardInterrupt
