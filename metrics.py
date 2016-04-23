"""
Code for various network metrics.
"""
from __future__ import division, print_function
import networkx as nx
import numpy as np


def paths_of_length(g, length):
    """
    Return list of all paths of a given length in a graph.
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