from __future__ import division, print_function
from copy import copy
from itertools import combinations
from itertools import product as cproduct
import networkx as nx
import numpy as np


def _calculate_softmax_probability(inputs):
    """
    Calculate softmax probabilities from inputs.
    :param inputs: input vector
    :return: vector of normalized probabilities
    """

    prob = np.exp(inputs)
    return prob / prob.sum()


class BasicWithAthAndTwoLevelStdp(object):
    """
    Most basic model with activation-triggered lingering hyperexcitability.
    """

    def __init__(self, th, w, g_x, t_x, rp, stdp_params):
        """
        :param th: input threshold above which node activates
        :param w: weight matrix
        :param g_x: hyperexcitability level
        :param t_x: hyperexcitability timescale
        :param rp: refractory period
        :param stdp_params: dictionary of stdp params:
            :param 'w_1': weak synaptic strength
            :param 'w_2': strong synaptic strength
            :param 'alpha_1': learning rate towards w_1
            :param 'alpha_2': learning rate towards w_2
        """

        self.th = th
        self.w = w
        self.g_x = g_x
        self.t_x = t_x
        self.rp = rp
        self.th = th

        self.n_nodes = w.shape[0]

        self.w_1 = 0 if stdp_params is None else stdp_params['w_1']
        self.w_2 = 0 if stdp_params is None else stdp_params['w_2']
        self.alpha_1 = 0 if stdp_params is None else stdp_params['alpha_1']
        self.alpha_2 = 0 if stdp_params is None else stdp_params['alpha_2']

    def update_weights(self, w, r_prev, r):
        """
        Update a weight matrix according to the STDP learning rate.
        Note that w is modified in place!
        :param w: previous weight matrix
        :param r_prev: previous unit activations
        :param r: current unit activations
        :return: updated weight matrix
        """

        if self.alpha_1 == self.alpha_2 == 0: return w

        # loop through all sequential activations
        for prev, curr in cproduct(r_prev.nonzero()[0], r.nonzero()[0]):

            if w[prev, curr]: w[prev, curr] += self.alpha_1*(self.w_1 - w[prev, curr])
            if w[curr, prev]: w[curr, prev] += self.alpha_2*(self.w_2 - w[curr, prev])

        return w

    def adjust_for_local_wta(self, v, r): return r

    def run(self, r_0, xc_0, drives, measure_w=None):
        """
        Run the network from a starting state by providing a stimulus.
        :param r_0: initial node activations
        :param xc_0: initial hyperexcitability states
        :param drives: stimuli for all nodes
        :param measure_w: function that takes in weight matrix as a single argument
            and outputs a quantity that will be stored in a list of measurements
        """

        rs = np.nan * np.zeros((self.n_nodes, len(drives)))
        xcs = np.nan * np.zeros((self.n_nodes, len(drives)))

        # set state for first time step
        r = r_0.copy()
        xc = xc_0.copy()
        rpc = np.zeros((self.n_nodes,))

        w = self.w.copy()
        w_measurements = []

        for t, drive in enumerate(drives):

            if t > 0:

                r_prev = r.copy()

                # calculate inputs and compare them to threshold
                x = (xc > 0).astype(float)
                v = w.dot(r) + drive + self.g_x*x
                r = (v > self.th).astype(int)
                # remove active nodes in refractory period
                r[rpc > 0] = 0

                w = self.update_weights(w, r_prev, r)

            else:
                v = np.ones(r.shape)

            # remove (nonexistent) WTA violations
            r = self.adjust_for_local_wta(v, r)

            # decrement hyperexcitabilities and refractory counter
            xc[xc > 0] -= 1
            rpc[rpc > 0] -= 1

            # make new active units hyperexcitable and refractory
            xc[r > 0] = self.t_x
            rpc[r > 0] = self.rp

            # store activities and hyperexcitabilities
            rs[:, t] = r
            xcs[:, t] = xc.copy()

            if measure_w is not None:
                w_measurements.append(measure_w(w))

        if measure_w is None:
            return rs.T, xcs.T
        else:
            return rs.T, xcs.T, w_measurements


class LocalWtaWithAthAndStdp(BasicWithAthAndTwoLevelStdp):

    def __init__(self, th, w, g_x, t_x, rp, stdp_params, wta_dist, wta_factor):

        super(self.__class__, self).__init__(th, w, g_x, t_x, rp, stdp_params)

        self.wta_dist = wta_dist
        self.wta_factor = wta_factor
        self.node_distances = np.zeros(w.shape)

        # use networkx to get shortest path length dict
        # and fill in shortest paths in the node distances matrix

        g = nx.Graph((self.w + self.w.T > 0).astype(int))

        for node_0, spls in nx.shortest_path_length(g).items():
            for node_1, spl in spls.items():

                self.node_distances[node_0, node_1] = spl
                self.node_distances[node_1, node_0] = spl

    def adjust_for_local_wta(self, v, r):
        """
        Ensure that no two nodes are active if they are <= self.wta_distance from each other
        :param v: node inputs
        :param r: candidate activation vector (prior to WTA correction)
        :return: corrected activation vector
        """

        active = list(r.nonzero()[0])

        # get all pairs of nodes that are too close together
        invalid_pairs = []
        for pair in combinations(active, 2):

            if 0 < self.node_distances[pair[0], pair[1]] <= self.wta_dist:
                invalid_pairs.append(pair)

        # turn off nodes with probability proportional to the sum of the inputs
        # of their potentially active neighbors until there are no more invalid pairs
        while invalid_pairs:

            # loop through list of all unique nodes
            input_sums = np.nan * np.zeros((len(active),))

            for ctr, node in enumerate(active):

                # find node's neighbors
                neighbors = [c for c in active if (node, c) in invalid_pairs or (c, node) in invalid_pairs]
                # store sum of neighbors' inputs
                input_sums[ctr] = v[neighbors].sum()

            probs = np.exp(input_sums * self.wta_factor)
            probs /= probs.sum()
            to_inactivate = np.random.choice(active, p=probs)

            active.pop(active.index(to_inactivate))
            invalid_pairs = [pair for pair in invalid_pairs if to_inactivate not in pair]

        if r.sum(): assert len(active) > 0

        r_adjusted = np.zeros(r.shape)
        r_adjusted[active] = 1

        return r_adjusted
