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
    :param refrac_per: refractory period or list of refractory periods for individual cells

    :param ws: dict of weight matrices for different synapse types
    """

    @staticmethod
    def update_conductances(gs, taus, ws, spikes, drives, dt):
        """
        Update conductances according to exponential ODE.
        """

        for syn in gs.keys():

            g, tau, w, drive = gs[syn], taus[syn], ws[syn], drives[syn]
            dg = (dt / tau) * (-g + (w.dot(spikes) + drive) * tau / dt)

            gs[syn] += dg

        return gs

    @staticmethod
    def update_voltages(vs, taus_m, gs, v_revs, v_rests, dt):
        """
        Update voltages according to exponential ODE.
        """

        inputs = np.array([g * (v_revs[syn] - vs) for syn, g in gs.items()])
        dv = (dt / taus_m) * (v_rests - vs + inputs.sum(0))

        return vs + dv

    @staticmethod
    def record_measurements(
            measurements, variables, t_ctr,
            spikes, voltages, refractory_counters, conductances):

        for variable in variables:

            if variable == 'spikes':
                measurements[variable][t_ctr, :] = spikes

            elif variable == 'voltages':
                measurements[variable][t_ctr, :] = voltages

            elif variable == 'refrac_ctrs':
                measurements[variable][t_ctr, :] = refractory_counters

            elif 'conductances' in variable:

                for key in measurements[variable].keys():
                    measurements[variable][key][t_ctr, :] = conductances[key]

        return measurements

    def __init__(
            self, v_rests, taus_m, taus_syn, v_revs_syn,
            v_ths, v_resets, refrac_pers, ws):

        self.v_rests = v_rests
        self.taus_m = taus_m
        self.taus_syn = taus_syn
        self.v_revs_syn = v_revs_syn

        self.v_ths = v_ths
        self.v_resets = v_resets

        self.ws = ws

        # extract some basic metadata
        self.n_cells = len(self.ws.values()[0])
        self.syns = self.taus_syn.keys()

        # allow refractory period to be specified for individual cells or not
        self.refrac_pers = np.array(refrac_pers)

    def run(self, initial_conditions, drives, dt, record=('spikes')):
        """
        Run a simulation

        :param initial_conditions: dict of initial voltages, conductances, and
            refractory periods
        :param drives: drives to each type of synapse at each neuron at each time point
        :param dt: integration time step
        :param record: tuple of variables to record, options are:
            spikes, voltages, refrac_ctrs, conductances
        :return: dictionary of measured variables at each time step
        """

        n_steps = np.max([drive.shape[0] for drive in drives.values()])

        # set initial conditions
        vs = initial_conditions['voltages']
        gs = {syn: initial_conditions['conductances'][syn] for syn in self.syns}
        rp_ctrs = initial_conditions['refrac_ctrs'] // dt
        spikes = (vs > self.v_ths).astype(float)

        # allocate space for variables to be measured
        measurements = {}

        for variable in record:

            if variable != 'conductances':
                measurements[variable] = np.zeros((n_steps + 1, self.n_cells))
            else:
                measurements[variable] = {
                    key: np.zeros((n_steps + 1, self.n_cells))
                    for key in self.syns
                }

        # record initial measurements
        self.record_measurements(measurements, record, 0,
            spikes, vs, rp_ctrs, gs)

        # run simulation
        for t_ctr in range(n_steps):

            # decrement nonzero refractory periods
            rp_ctrs[rp_ctrs > 0] -= 1

            # get drives for this time step
            drive = {syn: drives[syn][t_ctr] for syn in self.syns}

            # calculate conductances for all cells
            gs = self.update_conductances(gs, self.taus_syn, self.ws, spikes, drive, dt)

            # calculate voltage change for all cells
            vs = self.update_voltages(
                vs, self.taus_m, gs, self.v_revs_syn, self.v_rests, dt)

            # set voltage of refractory neurons to reset potential
            rp_mask = (rp_ctrs > 0) * (vs > self.v_resets)
            vs[rp_mask] = self.v_resets[rp_mask]

            # detect spikes, reset voltages, and set refractory periods
            spikes = vs > self.v_ths
            vs[spikes] = self.v_resets[spikes]
            rp_ctrs[spikes] = self.refrac_pers[spikes] // dt
            spikes = spikes.astype(float)

            # record desired variables

            self.record_measurements(measurements, record, t_ctr + 1,
                spikes, vs, rp_ctrs, gs)

        measurements['time'] = np.arange(n_steps + 1) * dt

        return measurements