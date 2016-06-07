from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from network import LIFExponentialSynapsesModel
from plot import set_fontsize


def single_neuron_driven_by_external_current_pulses(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        DRIVE_STRENGTHS, DRIVE_TIMES, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    drives = {syn: np.zeros((n_steps, 1)) for syn in syns}

    for syn, drive_times in DRIVE_TIMES.items():

        for t in drive_times:

            drives[syn][int(t / DT), 0] = DRIVE_STRENGTHS[syn]

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

    for syn in syns:

        axs[1].plot(measurements['time'], measurements['conductances'][syn][:, 0], lw=2)

    axs[1].legend([syn.upper() for syn in syns])

    axs[1].set_ylabel('conductances')

    axs[1].set_xlabel('time (s)')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def two_neuron_ei_circuit(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        W_IE, W_EI,
        DRIVE_STRENGTHS, DRIVE_TIMES, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE, COLORS):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    drives = {syn: np.zeros((n_steps, 2)) for syn in syns}

    for syn, drive_times in DRIVE_TIMES.items():

        for t in drive_times:

            drives[syn][int(t / DT), 0] = DRIVE_STRENGTHS[syn]

    # build network

    ws = {syn: np.array([[0, W_EI[syn]], [W_IE[syn], 0]]) for syn in syns}

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': np.array([V_REST, V_REST]),
        'conductances': {syn: np.array([0., 0.]) for syn in syns},
        'refrac_ctrs': np.array([0, 0]),
    }

    # set desired measurables

    record = ('voltages', 'spikes', 'conductances')

    # run simulation

    np.random.seed(SEED)

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)

    # MAKE PLOTS

    fig, axs = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    # plot voltage

    for voltages, color in zip(measurements['voltages'].T, COLORS):

        axs[0].plot(measurements['time'], voltages, color=color, lw=2)

    # plot spikes

    for neuron_ctr, (spikes, color) in enumerate(zip(measurements['spikes'].T, COLORS)):

        spike_times = measurements['time'][spikes.astype(bool)]
        ys = V_TH * np.ones((len(spike_times))) + 0.01

        axs[0].scatter(spike_times, ys, s=200, c=color, marker='*')

    axs[0].legend(['E', 'I'])

    axs[0].set_ylabel('voltage (V)')

    # plot conductances

    for ax_ctr, ax in enumerate(axs[1:]):

        for syn in syns:

            ax.plot(measurements['time'], measurements['conductances'][syn][:, ax_ctr], lw=2)

        ax.set_ylabel('conductances')

    axs[2].legend([syn.upper() for syn in syns])

    axs[2].set_xlabel('time (s)')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig