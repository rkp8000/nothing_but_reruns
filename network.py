from __future__ import division, print_function
from copy import copy
import numpy as np


class SoftmaxWTAWithLingeringHyperexcitability(object):
    """
    Most basic network model with activation-dependent excitability changes.

    In this network, exactly one node is allowed to be active at each time step. Which node becomes active
    is chosen with a probability that depends exponentially on its input (i.e., according to the softmax function).

    The possible inputs to the node at a current time step, which are weighted by their respective gains, are:
        upstream (from active node at last time step)
        lingering excitability (excitability changes are represented as inputs)
        external drive

    :param w: weight matrix (rows are targets, cols are sources)
    :param g_w: gain on upstream inputs
    :param g_x: gain on lingering hyperexcitability
    :param g_d: gain on external drive
    :param t_x: timescale of lingering hyperexcitability
    """

    def __init__(self, w, g_w, g_x, g_d, t_x):
        self.w = w
        self.g_w = g_w
        self.g_x = g_x
        self.g_d = g_d
        self.t_x = t_x

        self.n_nodes = w.shape[0]

    def run(self, r_0, xc_0, drives):
        """
        Run the network under a set of external drives.

        :param r_0: initial activation state
        :param xc_0: initial hyperexcitability counter
        :param drives: external drives at each time point
        :return: network activation at each time point
        """

        r = copy(r_0)
        xc = copy(xc_0)

        rs = []

        for drive in drives:

            # calculate inputs and softmax probabilities
            upstream = self.w.dot(r)
            excitability = (xc > 0).astype(float)

            all_input = self.g_w * upstream + self.g_x * excitability + self.g_d * drive

            prob = np.exp(all_input)
            prob /= prob.sum()

            # select active node
            active_node = np.random.choice(range(self.n_nodes), p=prob)
            r = np.zeros((self.n_nodes))
            r[active_node] = 1.

            rs.append(r)

            # decrement hyperexcitability counters
            xc[xc > 0] -= 1
            # turn on hyperexcitability of last active node
            xc[active_node] = self.t_x

        return np.array(rs)