import tensorflow as tf
import numpy as np
import time
from mpi4py import MPI
from collections import deque
from contextlib import contextmanager

from common.logger import Logger

from baselines.baselines_common import explained_variance, zipsame, dataset
import baselines.baselines_common.tf_util as U
from baselines.baselines_common import colorize
from baselines.baselines_common.mpi_adam import MpiAdam
from baselines.baselines_common.mpi_saver import MpiSaver
from baselines.baselines_common.cg import cg

from baselines.trajectories import traj_segment_generator, add_vtarg_and_adv


def learn(env, policy_func, args, *,
          timesteps_per_batch,  # what to train on
          max_kl, cg_iters,
          gamma, lam,  # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters=3):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(
        dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    policy_var_list = [v for v in all_var_list if v.name.split("/")[0].startswith("pi")]
    saver = MpiSaver(policy_var_list, log_prefix=args.logdir)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    gvp = tf.add_n([U.sum(g * tangent) for (g, tangent) in
                    zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function(
        [], [],
        updates=[tf.assign(oldv, newv)
                 for (oldv, newv) in
                 zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    saver.restore(restore_from=args.restore_actor_from)
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, args, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

    args.logdir = "{}/thread_{}".format(args.logdir, args.thread)
    logger = Logger(args.logdir)

    while time.time() - tstart < 86400 * args.max_train_days:
        # logger.log("********** Iteration %i ************" % iters_so_far)
        meanlosses = [0] * len(loss_names)
        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        segargs = seg["ob"], seg["ac"], seg["adv"]
        fvpargs = [arr[::5] for arr in segargs]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new()  # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*segargs)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            pass
        #     logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*segargs)))
                improve = surr - surrbefore
                # logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                # if not np.isfinite(meanlosses).all():
                #     logger.log("Got non-finite value of losses -- bad!")
                # elif kl > max_kl * 1.5:
                #     logger.log("violated KL constraint. shrinking step.")
                # elif improve < 0:
                #     logger.log("surrogate didn't improve. shrinking step.")
                # else:
                #     logger.log("Stepsize OK!")
                #     break
                stepsize *= .5
            else:
                # logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather(
                    (thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False,
                                                         batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        saver.sync()

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
