from collections import OrderedDict
from itertools import tee

import torch
import torch.nn as nn

from common.modules.LayerNorm import LayerNorm


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class LinearNet(nn.Module):
    def __init__(self, layers, activation=torch.nn.ELU,
                 layer_norm=False, linear_layer=nn.Linear):
        super(LinearNet, self).__init__()
        self.input_shape = layers[0]
        self.output_shape = layers[-1]

        if layer_norm:
            layer_fn = lambda layer: [
                ("linear_{}".format(layer[0]), linear_layer(layer[1][0], layer[1][1])),
                ("layer_norm_{}".format(layer[0]), LayerNorm(layer[1][1])),
                ("act_{}".format(layer[0]), activation())]
        else:
            layer_fn = lambda layer: [
                ("linear_{}".format(layer[0]), linear_layer(layer[1][0], layer[1][1])),
                ("act_{}".format(layer[0]), activation())]

        self.net = torch.nn.Sequential(
            OrderedDict([
                x for y in map(
                    lambda layer: layer_fn(layer),
                    enumerate(pairwise(layers))) for x in y]))

    def forward(self, x):
        x = self.net.forward(x)
        return x
