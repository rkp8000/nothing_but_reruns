"""
Code for various network metrics.
"""
from __future__ import division, print_function
import networkx as nx
import numpy as np
from scipy import linalg as sp_linalg
from scipy import stats


def paths_of_length(g, length):
    """
    Return list of all simple paths of a given length in a graph.
    :param g: networkx graph
    :param length: how long paths should be (number of nodes including start and end)
    :return: List of paths
    """
    paths = []

    for src in g.nodes():

        for targ in g.nodes():

            src_targ_paths = nx.all_simple_paths(g, src, targ, cutoff=length-1)
            paths.extend([tuple(path) for path in src_targ_paths if len(path) == length])

    return paths


def path_is_replayable(path, a):
    """
    Determine whether a path is replayable given an adjacency matrix. Note that in adjacency matrix
    rows are targets, cols are sources. This is the reverse of the networkx adjacency matrix.

    A path is defined as replayable if given the starting node as well as the set of other nodes
    in the path (but not their order), the path can be uniquely reconstructed from the graph
    connectivity.

    Specifically, a path is defined as replayable is given the subnetwork defined by the nodes in the
    path and their interconnections, no node in the path (except the last) has more than one node
    downstream of it.

    :param path: path to check (tuple)
    :param a: adjacency matrix (rows are targs, cols are sources)
    :return: True if path is replayable, False otherwise.
    """

    for k, src in enumerate(path[:-1]):

        for targ in path[k + 2:]:

            # return false if a downstream node is found
            if a[targ, src] == 1:

                return False

    return True


def replayable_paths(g, length, progress_ctr=0):
    """
    Return a list of all "replayable" paths in a graph. See documentation of "replayable" function for
    definition of replayability.

    :param g: graph
    :param length: length of paths to look at
    :return: List of replayable paths, list of nonreplayable paths
    """

    paths_all = paths_of_length(g, length)
    a = nx.adjacency_matrix(g).T

    paths_replayable = []
    paths_non_replayable = []

    for ctr, path in enumerate(paths_all):

        if progress_ctr:

            if ctr % progress_ctr == 0:

                print('{} paths checked'.format(ctr))

        if path_is_replayable(path, a):

            paths_replayable.append(path)

        else:

            paths_non_replayable.append(path)

    return paths_replayable, paths_non_replayable


def softmax_prob_from_weights(weights, gain):
    """
    Convert a weight matrix to a probabilistic transition matrix using a softmax rule.
    :param weights: weight matrix (rows are targs, cols are srcs)
    :param gain: scaling factor in softmax function (higher gain -> lower entropy)
    :return: transition matrix (rows are targs, cols are srcs), initial probabilities
    """

    # calculate transition probabilities p
    p_unnormed = np.exp(gain * weights)
    p = p_unnormed / p_unnormed.sum(0)

    # define p0 to be stationary distribution, given by principal eigenvector
    evals, evecs = np.linalg.eig(p)
    idxs_sorted = np.real(evals).argsort()[::-1]

    # normalize
    p0_unnormed = evecs[:, idxs_sorted[0]]
    p0 = np.real(p0_unnormed / p0_unnormed.sum())

    return p, p0


def occurrence_count(seq, states):
    """
    Count how many times each state occurs in a sequence.
    :param seq: discrete sequence
    :param states: set of states
    :return: array of counts (same length as states)
    """

    counts = np.zeros((len(states),), dtype=float)

    states_dict = {state: ctr for ctr, state in enumerate(states)}

    for el in seq:

        counts[states_dict[el]] += 1

    return counts


def transition_count(seq, states):
    """
    Count how many of each type of transition occur in a sequence.

    :param seq: discrete sequence
    :param states: set of states
    :return: transition count matrix (rows are "to", cols are "from")
    """

    transitions = np.zeros((len(states), len(states)), dtype=float)

    states_dict = {state: ctr for ctr, state in enumerate(states)}

    for el_to, el_from in zip(seq[1:], seq[:-1]):

        transitions[states_dict[el_to], states_dict[el_from]] += 1

    return transitions


def transition_dkl(t_0, t_1, base=2):
    """
    Calculate the DKL between two transition matrices (rows are "to", cols are "from").

    Since transition matrices give conditional distributions, we first calculate the joint transition
    distributions and then take the DKL between those two distributions.

    :param t_0: first transition matrix
    :param t_1: second transition matrix.
    :param base: base to calculate DKL in (default = 2)
    :return: DKL averaged over cols
    """

    t_joints = []

    for t_mat in [t_0, t_1]:

        # get stationary distribution

        evals, evecs = np.linalg.eig(t_mat)
        idxs_sorted = np.real(evals).argsort()[::-1]

        # normalize

        p_0_unnormed = evecs[:, idxs_sorted[0]]
        p_0 = np.real(p_0_unnormed / p_0_unnormed.sum())

        t_joints.append(t_mat * np.tile(p_0[:, None], (1, t_mat.shape[1])))

    return stats.entropy(t_joints[0].flatten(), t_joints[1].flatten(), base=base)


def gather_sequences(x, seq_len):
    """
    Convert a 1D array into a 2D array of consecutive sequences.

    E.g., gather_sequences(array([1, 2, 3, 4, 5]), 4) = array([[1, 2, 3, 4], [2, 3, 4, 5]])
    :param x:
    :return: 2D sequence array
    """

    return np.fliplr(sp_linalg.toeplitz(x, x[:seq_len]))[seq_len - 1:]