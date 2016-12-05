from __future__ import division, print_function
from itertools import product as cproduct


def zip_cproduct(z, c, order, **kwargs):
    zipped = zip(*[kwargs[k] for k in z])
    temp_0 = list(cproduct(*([zipped] + list([kwargs[k] for k in c]))))
    temp_1 = [ii[0] + ii[1:] for ii in temp_0]

    current_order = z + c
    temp_2 = [
        tuple([ii[current_order.index(jj)] for jj in order])
        for ii in temp_1
        ]

    return temp_2
