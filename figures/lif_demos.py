from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from network import LIFExponentialSynapsesModel
from plot import set_fontsize


def single_neuron_driven_by_external_current_pulses(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        DRIVE_STRENGTHS, DRIVE_STARTS, DRIVE_INTERVALS, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    drives = {syn: np.zeros((n_steps, 1)) for syn in syns}

    for syn in syns:
        drive_start = int(DRIVE_STARTS[syn] / DT)
        drive_interval = int(DRIVE_INTERVALS[syn] / DT)
        drive_strength = DRIVE_STRENGTHS[syn]

        drives[syn][drive_start::drive_interval] = drive_strength

    # build network

    ws = {syn: np.array([[0.]]) for syn in syns}

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': np.array([V_REST]),
        'conductances': {syn: np.array([0.]) for syn in syns},
        'refrac_ctrs': np.array([0]),
    }

    # set desired measurables

    record = ('voltages', 'spikes', 'conductances')

    # run simulation

    np.random.seed(SEED)

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)


    # MAKE PLOTS

    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    # plot voltage

    axs[0].plot(measurements['time'], measurements['voltages'][:, 0], color='k', lw=2)

    axs[0].set_ylabel('voltage (V)')

    # plot conductances

    axs[1].plot(measurements['time'], measurements['conductances'].values()[0], color='g', lw=2)

    axs[1].set_ylabel('conductances')

    axs[1].set_xlabel('time (s)')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig