from __future__ import division
import networkx as nx
import numpy as np


def erdos_renyi(n_nodes, p_connect):
    """
    Build a directed Erdos-Renyi graph.

    :param n_nodes: number of nodes
    :param p_connect: connection probability
    :return: graph
    """

    return nx.erdos_renyi_graph(n=n_nodes, p=p_connect, directed=True)