from __future__ import division, print_function
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