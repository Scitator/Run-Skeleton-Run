from mpi4py import MPI
import baselines.baselines_common.tf_util as U
import tensorflow as tf


class MpiSaver(object):
    def __init__(self, var_list=None, *,
                 comm=None,
                 log_prefix="/tmp"):
        self.var_list = var_list
        self.t = 0

        self.saver = tf.train.Saver(
            var_list=var_list,
            max_to_keep=100,
            keep_checkpoint_every_n_hours=0.25,
            pad_step_number=True,
            save_relative_paths=True)
        self.log_prefix = log_prefix

        self.comm = MPI.COMM_WORLD if comm is None else comm

    def restore(self, restore_from=None):
        if restore_from is not None:
            self.saver.restore(U.get_session(), restore_from)
            self.t += int(restore_from.split("-")[-1])
        self.sync()

    def sync(self):
        if self.comm.Get_rank() == 0:  # this is root
            self.saver.save(
                U.get_session(),
                "{}/model.ckpt".format(self.log_prefix),
                global_step=self.t)
            self.t += 1
