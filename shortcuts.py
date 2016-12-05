from __future__ import division, print_function
from itertools import product as cproduct


def zip_cproduct(z, c, **kwargs):
    zipped = zip(*[kwargs[k] for k in z])
    cproducted = list(cproduct(*([zipped] + list([kwargs[k] for k in c]))))

    return [ii[0] + ii[1:] for ii in cproducted]