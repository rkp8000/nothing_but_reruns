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


class LIFExponentialSynapsesModel(object):
    """
    Model network composed of leaky integrate-and-fire neurons with exponential
    synapses. All parameters are in SI units except weights, which are in units
    of conductance relative to the leak conductance.

    :param v_rest: membrane resting potential
    :param tau_m: membrane time constant
    :param taus_syn: dict of synaptic time constants
    :param v_revs_syn: dict of synaptic reveresal potentials

    :param v_th: threshold potential
    :param v_reset: reset potential
    :param refractory_period: refractory period

    :param ws: dict of weight matrices for different synapse types
    """

    def __init__(
            self, v_rest, tau_m, taus_syn, v_revs_syn,
            v_th, v_reset, refractory_period, ws):

        self.v_rest = v_rest
        self.tau_m = tau_m
        self.taus_syn = taus_syn
        self.v_revs_syn = v_revs_syn

        self.v_th = v_th
        self.v_reset = v_reset

        self.refractory_period = refractory_period
        self.ws = ws

        # extract some basic metadata

        self.n_cells = len(self.ws.items()[1])
        self.synapses = self.taus_syn.keys()

    def run(self, initial_conditions, drives, dt, record=('spikes')):
        """
        Run a simulation

        :param initial_conditions:
        :param drives:
        :param dt:
        :param record:
        :return:
        """

        n_steps = np.max([drive.shape[0] for k, drive in drives.items()])

        rp_dt = self.refractory_period // dt

        # set initial conditions

        vs = initial_conditions['voltages']
        gs = {key: initial_conditions['conductances'][key] for key in self.synapses}
        rp_ctrs = initial_conditions['refractorinesses'] // dt
        spikes = np.zeros((self.n_cells,))

        # allocate space for variables and run simulation

        measurements = {key: np.zeros((n_steps, self.n_cells))}

        for t_ctr in range(n_steps):

            # calculate conductances for all cells



            # calculate voltage change for all cells

            # record desired variables

            for variable in record:

                if variable == 'spikes':

                    pass

                elif variable == 'voltages':

                    pass

                elif variable == 'conductances':

                    pass

                elif variable == 'refractory_counters':

                    pass