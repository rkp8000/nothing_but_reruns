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
        W_IE, W_EI, W_II,
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

    ws = {syn: np.array([[0, W_EI[syn]], [W_IE[syn], W_II[syn]]]) for syn in syns}

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


def three_neuron_eei_circuit(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        W_EE, W_IE, W_EI, W_II,
        DRIVE_STRENGTHS, DRIVE_TIMES, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE, COLORS):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    drives = {syn: np.zeros((n_steps, 3)) for syn in syns}

    for syn, drive_times in DRIVE_TIMES.items():

        for t in drive_times:

            drives[syn][int(t / DT), 0] = DRIVE_STRENGTHS[syn]

    # build network

    ws = {syn: np.array([
        [        0,         0, W_EI[syn]],
        [W_EE[syn],         0, W_EI[syn]],
        [W_IE[syn], W_IE[syn], W_II[syn]],
    ]) for syn in syns}

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': np.array([V_REST, V_REST, V_REST]),
        'conductances': {syn: np.array([0., 0., 0.]) for syn in syns},
        'refrac_ctrs': np.array([0, 0, 0]),
    }

    # set desired measurables

    record = ('voltages', 'spikes', 'conductances')

    # run simulation

    np.random.seed(SEED)

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)

    # MAKE PLOTS

    fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    # plot voltage

    for voltages, color in zip(measurements['voltages'].T, COLORS):

        axs[0].plot(measurements['time'], voltages, color=color, lw=2)

    # plot spikes

    for neuron_ctr, (spikes, color) in enumerate(zip(measurements['spikes'].T, COLORS)):
        spike_times = measurements['time'][spikes.astype(bool)]
        ys = V_TH * np.ones((len(spike_times))) + 0.01

        axs[0].scatter(spike_times, ys, s=200, c=color, marker='*')

    axs[0].legend(['E_us', 'E_ds', 'I'])

    axs[0].set_ylabel('voltage (V)')

    # plot conductances

    for ax_ctr, ax in enumerate(axs[1:]):

        for syn in syns:
            ax.plot(measurements['time'], measurements['conductances'][syn][:, ax_ctr], lw=2)

        ax.set_ylabel('conductances')

    axs[3].legend([syn.upper() for syn in syns])

    axs[3].set_xlabel('time (s)')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def multineuron_chain_ei_circuit(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        N_E_NEURONS,
        W_EE, W_IE, W_EI, W_II,
        DRIVE_STRENGTHS, DRIVE_TIMES, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE, COLORS):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    drives = {syn: np.zeros((n_steps, N_E_NEURONS + 1)) for syn in syns}

    for syn, drive_times in DRIVE_TIMES.items():

        for t in drive_times:

            drives[syn][int(t / DT), 0] = DRIVE_STRENGTHS[syn]

    # build network

    ws = {syn: np.zeros((N_E_NEURONS + 1, N_E_NEURONS + 1)) for syn in syns}

    for syn in syns:

        ws[syn][range(1, N_E_NEURONS), range(0, N_E_NEURONS - 1)] = W_EE[syn]
        ws[syn][:-1, -1] = W_EI[syn]
        ws[syn][-1, :-1] = W_IE[syn]
        ws[syn][-1, -1] = W_II[syn]

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': V_REST * np.ones((N_E_NEURONS + 1,)),
        'conductances': {syn: np.zeros((N_E_NEURONS + 1,)) for syn in syns},
        'refrac_ctrs': np.zeros((N_E_NEURONS + 1,)),
    }

    # set desired measurables

    record = ('voltages', 'spikes', 'conductances')

    # run simulation

    np.random.seed(SEED)

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)

    # MAKE PLOTS

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    # plot voltage

    for voltages, color in zip(measurements['voltages'].T, COLORS):

        ax.plot(measurements['time'], voltages, color=color, lw=2)

    # plot spikes

    for neuron_ctr, (spikes, color) in enumerate(zip(measurements['spikes'].T, COLORS)):

        spike_times = measurements['time'][spikes.astype(bool)]
        ys = V_TH * np.ones((len(spike_times))) + 0.01

        ax.scatter(spike_times, ys, s=200, c=color, marker='*')

    ax.legend(['E_{}'.format(ctr) for ctr in range(N_E_NEURONS)] + ['I'])

    ax.set_xlabel('time (s)')
    ax.set_ylabel('voltage (V)')

    set_fontsize(ax, FONT_SIZE)

    return fig


def self_sustaining_excitation_among_e_cells(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        N_NEURONS,
        W_EE, P_CONNECT,
        DRIVE_STRENGTHS, DRIVE_TIMES, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    drives = {syn: np.zeros((n_steps, N_NEURONS)) for syn in syns}

    for syn, drive_times in DRIVE_TIMES.items():

        for t in drive_times:

            drives[syn][int(t / DT), :] = DRIVE_STRENGTHS[syn]

    # build network

    ws = {syn: np.zeros((N_NEURONS, N_NEURONS)) for syn in syns}

    np.random.seed(SEED)

    for syn in syns:

        cxn_mask = np.random.rand(N_NEURONS, N_NEURONS) < P_CONNECT[syn]
        np.fill_diagonal(cxn_mask, 0)

        ws[syn][cxn_mask] = W_EE[syn]

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': V_REST * np.ones((N_NEURONS,)),
        'conductances': {syn: np.zeros((N_NEURONS,)) for syn in syns},
        'refrac_ctrs': np.zeros((N_NEURONS,)),
    }

    # set desired measurables

    record = ('voltages', 'spikes', 'conductances')

    # run simulation

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)

    # MAKE PLOTS

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    # plot voltage

    for voltages in measurements['voltages'].T:

        ax.plot(measurements['time'], voltages, color='k', lw=2)

    # plot spikes

    for neuron_ctr, spikes in enumerate(measurements['spikes'].T):

        spike_times = measurements['time'][spikes.astype(bool)]
        ys = V_TH * np.ones((len(spike_times))) + 0.01

        ax.scatter(spike_times, ys, s=200, c='k', marker='*')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('voltage (V)')

    set_fontsize(ax, FONT_SIZE)

    return fig


def hyperexcitable_downstream_neurons_via_input_barrage(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M, TAUS_SYN, V_REVS_SYN,
        CHAIN_LENGTH,
        W_EE, W_IE, W_EI, W_II,
        BARRAGE_FREQS, BARRAGE_STRENGTHS,
        DRIVE_STRENGTHS, DRIVE_TIMES, SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE, COLORS):

    syns = TAUS_SYN.keys()

    # build drive arrays

    n_steps = int(SIM_DURATION / DT)
    n_cells = CHAIN_LENGTH + 3
    drives = {syn: np.zeros((n_steps, n_cells)) for syn in syns}

    # external drive

    for syn, drive_times in DRIVE_TIMES.items():

        for t in drive_times:

            drives[syn][int(t / DT), 0] = DRIVE_STRENGTHS[syn]

    # barrage drive

    for syn, freq in BARRAGE_FREQS.items():

        mean_rate = freq * DT

        barrage = np.random.poisson(mean_rate, (n_steps,)) * BARRAGE_STRENGTHS[syn]

        drives[syn][:, -2] += barrage

    # build network

    ws = {syn: np.zeros((n_cells, n_cells)) for syn in syns}

    for syn in syns:

        ws[syn][range(1, CHAIN_LENGTH), range(0, CHAIN_LENGTH - 1)] = W_EE[syn]
        ws[syn][-3:-1, -4] = W_EE[syn]
        ws[syn][-1, :-1] = W_IE[syn]
        ws[syn][:-1, -1] = W_EI[syn]
        ws[syn][-1, -1] = W_II[syn]

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': V_REST * np.ones((n_cells,)),
        'conductances': {syn: np.zeros((n_cells,)) for syn in syns},
        'refrac_ctrs': np.zeros((n_cells,)),
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

    for cell_ctr, (spikes, color) in enumerate(zip(measurements['spikes'].T, COLORS)):

        spike_times = measurements['time'][spikes.astype(bool)]
        ys = V_TH * np.ones((len(spike_times))) + 0.01 + 0.005 * cell_ctr

        axs[0].scatter(spike_times, ys, s=200, c=color, marker='*')

    axs[0].set_ylabel('voltage (V)')

    axs[0].legend(['E_{}'.format(ctr) for ctr in range(n_cells - 1)] + ['I'])

    for cell_ctr, ax in zip([-3, -2], axs[1:]):

        for syn in syns:

            ax.plot(measurements['time'], measurements['conductances'][syn][:, cell_ctr], lw=2)

        ax.set_ylabel('conductances')

    axs[-1].legend([syn.upper() for syn in syns])

    axs[-1].set_xlabel('time (s)')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig