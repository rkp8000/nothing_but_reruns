from __future__ import division, print_function
from copy import copy
import numpy as np


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


class RateBasedModel(object):
    """
    Basic rate-based model.
    """

    def __init__(self, taus, v_rests, v_ths, gains, noises, w):

        assert isinstance(taus, np.ndarray)
        assert isinstance(v_rests, np.ndarray)
        assert isinstance(v_ths, np.ndarray)
        assert isinstance(gains, np.ndarray)
        assert isinstance(noises, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert w.shape[0] == w.shape[1]

        self.taus = taus
        self.v_rests = v_rests
        self.v_ths = v_ths
        self.gains = gains
        self.noises = noises
        self.w = w

        self.n_nodes = w.shape[0]

    def rate_from_voltage(self, v):

        return sigmoid(self.gains * (v - self.v_ths))

    def run(self, v_0s, drives, dt):

        n_time_steps = len(drives)

        vs = np.nan * np.zeros((1 + n_time_steps, self.n_nodes))
        rs = np.nan * np.zeros((1 + n_time_steps, self.n_nodes))

        vs[0, :] = v_0s
        rs[0, :] = self.rate_from_voltage(v_0s)

        for t_ctr, drive in enumerate(drives):

            # calculate change in voltage

            decay = -(vs[t_ctr, :] - self.v_rests)
            recurrent = self.w.dot(rs[t_ctr, :])
            noise = np.random.normal(0, self.noises)
            dv = (dt / self.taus) * (decay + recurrent + noise + drive)

            vs[t_ctr + 1, :] = vs[t_ctr, :] + dv
            rs[t_ctr + 1, :] = self.rate_from_voltage(vs[t_ctr + 1, :])

        return vs[1:], rs[1:]
