"""
Code for various network metrics.
"""
from __future__ import division, print_function
import networkx as nx
import numpy as np


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