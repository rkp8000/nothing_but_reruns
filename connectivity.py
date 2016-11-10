from __future__ import division, print_function
import networkx as nx
import numpy as np


def feed_forward_grid(shape, spread):
    """
    Construct a weight matrix for nodes arranged in a square grid, such that each node "feeds forward"
    with a connection to a number of nodes (determined by the "spread" parameter) in the column to its right.

    :param shape: grid shape (n_rows, n_cols)
    :param spread: lateral spread of feed forwardness; options:
        1: only one downstream node
        2: three downstream nodes
        3: five downstream nodes
        etc.
    :return: connectivity matrix (rows are targs, cols are sources)
    """

    n_nodes = shape[0] * shape[1]
    w = np.zeros((n_nodes, n_nodes))

    for row in range(shape[0]):

        for col in range(shape[1] - 1):

            # get multi indexes of downstream nodes

            downstream_multis = []

            for spread_ctr in range(spread):

                downstream_multis.append([(row + spread_ctr) % shape[0], col + 1])
                downstream_multis.append([(row - spread_ctr) % shape[0], col + 1])

            # convert to flat indexes

            downstream_flats = np.unique(np.ravel_multi_index(np.transpose(downstream_multis), shape))

            # add connections

            src = np.ravel_multi_index((row, col), shape)

            for targ in downstream_flats:

                w[targ, src] = 1

    return w


def er_directed(n_nodes, p_connect):
    """
    Construct a directed Erdos-Renyi network (with binary connection weights). This is just a wrapper
    around networkx's erdos_renyi_network function.

    :param n_nodes: number of nodes
    :param p_connect: connection probability
    :return: weight matrix (rows are targs, cols are srcs)
    """

    g = nx.erdos_renyi_graph(n_nodes, p_connect, directed=True)

    return np.array(nx.adjacency_matrix(g).T.todense())


def er_directed_nary(n_nodes, p_connect, strengths, p_strengths):
    """
    Construct a directed Erdos-Renyi network (with binary connection weights).

    :param n_nodes: number of nodes
    :param p_connect: connection probability
    :param strengths: list of strengths that weights can take
    :param p_strengths: probabilities that connections take on each strength
    :return: weight matrix (rows are targs, cols are srcs)
    """

    w = (np.random.rand(n_nodes, n_nodes) < p_connect).astype(float)
    np.fill_diagonal(w, 0)

    w[w > 0] = np.random.choice(strengths, size=(w.sum(),), p=p_strengths)

    return w


def basic_adlib(principal_connectivity_mask, w_pp, w_mp, w_pm, w_mm):
    """
    Build the connectivity for a basic ADLIB (activation-dependent lingering increases in baseline) network.

    :param primary_connectivity_mask: binary connections among principal nodes
    :param w_pp: connection strength among principal nodes
    :param w_mp: connection strength from principal to memory nodes
    :param w_pm: connection strength from memory to principal nodes
    :param w_mm: self connection strength for memory nodes
    :return: weight matrix
    """

    n_nodes = principal_connectivity_mask.shape[0]

    w_principal = principal_connectivity_mask.astype(float) * w_pp

    w_left = np.concatenate([w_principal, w_mp * np.eye(n_nodes)], axis=0)
    w_right = np.concatenate([w_pm * np.eye(n_nodes), w_mm * np.eye(n_nodes)], axis=0)

    return np.concatenate([w_left, w_right], axis=1)


def hexagonal_lattice(d):
    """
    Create a connectivity matrix corresponding to a hexagonal lattice with bidirectional
    connections between adjacent nodes.
    :param d: dimension of grid (length of outer hexagonal edge)
    :return: binary weight matrix
    """

    # make nodes
    nodes = []

    # top d rows
    for row_ctr in range(d):

        col_key_start = d - row_ctr - 1

        for col_ctr in range(d + row_ctr):

            row_key = row_ctr
            col_key = 2*col_ctr + col_key_start
            nodes.append((row_key, col_key))

    # bottom d - 1 rows
    for row_ctr in range(1, d):
        col_key_start = row_ctr

        for col_ctr in range(2 * d - row_ctr - 1):

            row_key = d + row_ctr - 1
            col_key = 2*col_ctr + col_key_start
            nodes.append((row_key, col_key))

    n_nodes = len(nodes)
    assert n_nodes == len(set(nodes))

    # make connectivity matrix
    w = np.zeros((n_nodes, n_nodes))

    for node in nodes:

        # get node's targets
        targs = []

        # prev row
        targs.append((node[0] - 1, node[1] - 1))
        targs.append((node[0] - 1, node[1] + 1))

        # same row
        targs.append((node[0], node[1] - 2))
        targs.append((node[0], node[1] + 2))

        # next row
        targs.append((node[0] + 1, node[1] - 1))
        targs.append((node[0] + 1, node[1] + 1))

        # remove connections to nonexisting nodes
        targs = [targ for targ in targs if targ in nodes]

        # add connections to w
        src_idx = nodes.index(node)

        for targ in targs:

            targ_idx = nodes.index(targ)
            w[targ_idx, src_idx] = 1

    # normalize node indexes so that (0, 0) is at center
    for node_ctr, node in enumerate(nodes):
        nodes[node_ctr] = (node[0] - d + 1, node[1] - 2*d + 2)

    return w, nodes
