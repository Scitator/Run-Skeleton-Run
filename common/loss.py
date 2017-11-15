import numpy as np
import torch
import torch.nn as nn


def create_linear_decay_fn(initial_value, final_value, max_step):
    def decay_fn(step):
        relative = 1. - step / max_step
        return initial_value * relative + final_value * (1. - relative)

    return decay_fn


def create_cycle_decay_fn(initial_value, final_value, cycle_len, num_cycles):
    max_step = cycle_len * num_cycles

    def decay_fn(step):
        relative = 1. - step / max_step
        relative_cosine = 0.5 * (np.cos(np.pi * np.mod(step, cycle_len) / cycle_len) + 1.0)
        return relative_cosine * (initial_value - final_value) * relative + final_value

    return decay_fn


def create_decay_fn(decay_type, **kwargs):
    if decay_type == "linear":
        return create_linear_decay_fn(**kwargs)
    elif decay_type == "cycle":
        return create_cycle_decay_fn(**kwargs)
    else:
        raise NotImplementedError()


class QuadricLinearLoss(nn.Module):
    def __init__(self, clip_delta):
        super(QuadricLinearLoss, self).__init__()
        self.clip_delta = clip_delta

    def forward(self, y_pred, y_true, weights):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        loss = torch.mean(loss * weights)
        return loss

losses = {
    "mse": nn.MSELoss,
    "quadric-linear": QuadricLinearLoss
}


def create_loss(args):
    if args.loss_type == "mse":
        return nn.MSELoss()
    elif args.loss_type == "quadric-linear":
        return QuadricLinearLoss(clip_delta=args.clip_delta)
    else:
        raise NotImplementedError()
