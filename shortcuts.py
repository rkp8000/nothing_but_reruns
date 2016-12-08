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
    idxs_reordered = [original.index(x) for x in reordered]

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
