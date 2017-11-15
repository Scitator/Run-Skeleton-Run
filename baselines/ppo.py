import tensorflow as tf
import numpy as np
import time
from mpi4py import MPI
from collections import deque
from contextlib import contextmanager

from common.logger import Logger
from baselines.baselines_common import Dataset, explained_variance, fmt_row, zipsame
import baselines.baselines_common.tf_util as U
from baselines.baselines_common.mpi_adam import MpiAdam
from baselines.baselines_common.mpi_saver import MpiSaver
from baselines.baselines_common.mpi_moments import mpi_moments

from baselines.trajectories import traj_segment_generator, add_vtarg_and_adv


def learn(env, policy_func, args, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          adam_epsilon=1e-5,
          schedule='constant'):  # annealing for stepsize parameters (epsilon and adam),
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32,
                           shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                             losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    policy_var_list = [v for v in var_list if v.name.split("/")[0].startswith("pi")]
    saver = MpiSaver(policy_var_list, log_prefix=args.logdir)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(),
                                                            pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    saver.restore(restore_from=args.restore_actor_from)
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, args, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    # max_timesteps = 1e10
    cur_lrmult = 1.0

    args.logdir = "{}/thread_{}".format(args.logdir, args.thread)
    logger = Logger(args.logdir)

    while time.time() - tstart < 86400 * args.max_train_days:
        # if schedule == 'constant':
        #     cur_lrmult = 1.0
        # elif schedule == 'linear':
        #     cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        # else:
        #     raise NotImplementedError

        # logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        # logger.log("Optimizing...")
        # logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"],
                                            batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            # logger.log(fmt_row(13, np.mean(losses, axis=0)))

        saver.sync()
        # logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                       cur_lrmult)
            losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        # logger.log(fmt_row(13, meanlosses))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        # Logging
        logger.scalar_summary("episodes", len(lens), iters_so_far)

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.scalar_summary(lossname, lossval, episodes_so_far)

        logger.scalar_summary("ev_tdlam_before", explained_variance(vpredbefore, tdlamret), episodes_so_far)

        logger.scalar_summary("step", np.mean(lenbuffer), episodes_so_far)
        logger.scalar_summary("reward", np.mean(rewbuffer), episodes_so_far)
        logger.scalar_summary("best reward", np.max(rewbuffer), episodes_so_far)

        elapsed_time = time.time() - tstart

        logger.scalar_summary(
            "episode per minute",
            episodes_so_far / elapsed_time * 60,
            episodes_so_far)
        logger.scalar_summary(
            "step per second",
            timesteps_so_far / elapsed_time,
            episodes_so_far)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
