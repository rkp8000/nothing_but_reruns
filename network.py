from __future__ import division, print_function
from copy import copy
import numpy as np


def _calculate_softmax_probability(inputs):
    """
    Calculate softmax probabilities from inputs.
    :param inputs: input vector
    :return: vector of normalized probabilities
    """

    prob = np.exp(inputs)
    return prob / prob.sum()


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

    def calculate_inputs(self, r, xc, drive):
        """
        Calculate vector of inputs to network given previous activation state, hyperexcitability counter, and drive.
        :param r: activation state at last time step
        :param xc: hyperexcitability counter
        :param drive: external drive
        :return: vector of inputs
        """

        upstream = self.w.dot(r)
        excitability = (xc > 0).astype(float)

        return self.g_w * upstream + self.g_x * excitability + self.g_d * drive

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
            inputs = self.calculate_inputs(r, xc, drive)
            prob = _calculate_softmax_probability(inputs)

            # select active node
            active_node = np.random.choice(range(self.n_nodes), p=prob)
            r = np.zeros((self.n_nodes,))
            r[active_node] = 1.

            rs.append(r)

            # decrement hyperexcitability counters
            xc[xc > 0] -= 1
            # turn on hyperexcitability of last active node
            xc[active_node] = self.t_x

        return np.array(rs)

    def sequence_probability(self, seq, r_0, xc_0, drives):
        """
        Calculate the probability of the network producing a sequence given an initial state and
        set of drives.
        :param seq: sequence to be produced (1-D array)
        :param r_0, xc_0, drives: see method "run"
        :return: probability of network producing sequence
        """

        p_seq = 1

        r = copy(r_0)
        xc = copy(xc_0)

        for candidate_node, drive in zip(seq, drives):

            # calculate inputs and softmax probabilities
            inputs = self.calculate_inputs(r, xc, drive)
            prob = _calculate_softmax_probability(inputs)

            # get probability of candidate node activating
            p_seq *= prob[candidate_node]

            # update activations and hyperexcitability counters as if the candidate node had become active,
            # so that the next probability can be calculated conditioned on previous part of the sequence
            r = np.zeros((self.n_nodes,))
            r[candidate_node] = 1.

            # decrement hyperexcitability counters
            xc[xc > 0] -= 1
            # turn on hyperexcitability of last active node
            xc[candidate_node] = self.t_x

        return p_seq