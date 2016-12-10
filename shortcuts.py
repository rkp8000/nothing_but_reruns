from __future__ import division, print_function
from itertools import product as cproduct
import numpy as np


def make_drive_seq(seq, nodes, start, shape):
    """
    Convert a sequence of nodes to a logical array.
    :param seq:
    :param nodes:
    :param start:
    :param shape:
    :return:
    """
    drives = np.zeros(shape)

    for ctr, node in enumerate(seq):
        drives[start + ctr, nodes.index(node)] = 1

    return drives


def reorder_idxs(original, first):
    """
    Reorder an array so that a specific ordered set of elements appear first.
    Return the reordered array and the indexes that were used to do the reordering.
    """
    if len(set(original)) != len(original):
        raise Exception('original array must not contain repeated element')
    if len(set(first)) != len(first):
        raise Exception('first elements must not contain repeated element')
    if np.any([x not in original for x in first]):
        raise Exception('first elements contain unknown element')

    reordered = list(first) + [x for x in original if x not in first]
    idxs_reordered = np.array([original.index(x) for x in reordered]).astype(int)

    return reordered, idxs_reordered


def zip_cproduct(z, c, order, kwargs):
    zipped = zip(*[kwargs[k] for k in z])
    temp_0 = list(cproduct(*([zipped] + list([kwargs[k] for k in c]))))
    temp_1 = [ii[0] + ii[1:] for ii in temp_0]

    current_order = z + c
    temp_2 = [
        tuple([ii[current_order.index(jj)] for jj in order])
        for ii in temp_1
        ]

    return temp_2


def get_stationary_distribution(trs):
    """
    Return stationary distribution given a transition matrix.
    :param trs:
    :return:
    """

    evs, evecs = np.linalg.eig(trs)

    idx = np.argmin(np.abs(np.abs(evs) - 1))
    if evs[idx] < 1: evecs *= -1

    p_0 = np.real(evecs[:, idx])
    p_0 /= p_0.sum()

    return p_0


def sample_markov_chain(p_0, trs, l):
    """
    Sample a sequence from a Markov chain.
    :param p_0:
    :param trs:
    :param l:
    :return:
    """
    nodes = np.arange(len(p_0))
    seq = [np.random.choice(nodes, p=p_0)]

    for _ in range(l-1):
        seq.append(np.random.choice(nodes, p=trs[:, seq[-1]]))

    return seq

