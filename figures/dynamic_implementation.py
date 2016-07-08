"""
Figures demonstrating dynamical systems implementation of replay.
"""
from __future__ import division, print_function
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import connectivity
import network
from plot import multivariate_same_axis
from plot import set_fontsize
from plot import firing_rate_heat_map


def sigmoid(x):

        return 1 / (1 + np.exp(-x))


def _build_tree_structure_demo_connectivity(
        branch_length, w_pp, w_mp, w_pm, w_mm):

    cc = np.concatenate

    # calculate each branch

    idx_offset = 3 * branch_length

    branch_0_0 = 2 * np.arange(branch_length + 1)
    branch_0_1 = np.arange(2 * branch_length + 1, 3 * branch_length + 1)

    branch_0 = cc([branch_0_0, branch_0_1])

    branch_2_0 = 2 * np.arange(branch_length + 1)
    branch_2_1 = 2 * np.arange(branch_length)[::-1] + 1

    branch_2 = cc([branch_2_0, branch_2_1])

    branches = [branch_0, -branch_0, branch_2, -branch_2]
    branches = [branch + idx_offset for branch in branches]

    # set up network

    n_principal_nodes = 6 * branch_length + 1

    # make list of all principal-to-principal connections

    cxns_pp = []

    for branch in branches:

        cxns = np.transpose([branch[1:], branch[:-1]])
        cxns_pp.extend([tuple(cxn) for cxn in cxns])

    # make principal-to-principal connection mask

    principal_mask = np.zeros((n_principal_nodes, n_principal_nodes))

    for cxn in cxns_pp:
        principal_mask[cxn] = 1

    # build final weight matrix

    w = connectivity.basic_adlib(
        principal_mask, w_pp=w_pp, w_mp=w_mp, w_pm=w_pm, w_mm=w_mm)

    return w, branches


def _tree_structure_replay_demo_simulation(
        SEED,
        TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        DRIVE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_PULSE_START,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT):
    """
    Demo sequential replay in a network with a basic tree-structure.

    There are always 4 unique paths from the root of the tree to the ends
    of the branches.

    The network with branch length 3 has the following tree structure:

    * -< * -< * -< * -< * -< * -< *
     \              \
     \               -< * -< * -< *
     \
      -< * -< * -< * -< * -< * -< *
                    \
                     -< * -< * -< *
    :param SEED: RNG seed
    :param TAU: single node time constant
    :param V_REST: resting potential
    :param V_TH: threshold potential
    :param GAIN: gain
    :param NOISE: noise level
    :param DT: integration time interval
    :param BRANCH_LENGTH: how many nodes per branch
    :param W_PP: principal-to-principal weight
    :param W_MP: principal-to-memory weight
    :param W_PM: memory-to-principal weight
    :param W_MM: memory-to-memory weight
    :param DRIVE_START: time of starting pulse train drives
    :param PULSE_DURATION: pulse duration
    :param INTER_PULSE_INTERVAL: time between consecutive pulse starts
    :param PULSE_HEIGHT: pulse height
    :param BRANCH_ORDER: which order to stimulate branches in
    :param INTER_TRAIN_INTERVAL: interval between pulse train starts
    :param REPLAY_PULSE_START: time to trigger the replay pulse, relative to train start
    :param RESET_PULSE_START: time of reset pulse, relative to train start
    :param RESET_PULSE_DURATION: duration of reset pulse
    :param RESET_PULSE_HEIGHT: height of reset pulse

    :return: n_nodes, drives, vs, rs
    """

    def to_time_steps(interval):

        return int(interval / DT)

    np.random.seed(SEED)

    w, branches = _build_tree_structure_demo_connectivity(
        branch_length=BRANCH_LENGTH,
        w_pp=W_PP, w_mp=W_MP, w_pm=W_PM, w_mm=W_MM)

    n_nodes = w.shape[0]

    # build final network

    ntwk = network.RateBasedModel(
        taus=TAU * np.ones((n_nodes,)),
        v_rests=V_REST * np.ones((n_nodes,)),
        v_ths=V_TH * np.ones((n_nodes,)),
        gains=GAIN * np.ones((n_nodes,)),
        noises=NOISE * np.ones((n_nodes,)),
        w=w)

    # build drive sequences

    drives = [np.zeros((to_time_steps(DRIVE_START), n_nodes))]

    for branch_idx in BRANCH_ORDER:

        drives_partial = np.zeros((to_time_steps(INTER_TRAIN_INTERVAL), n_nodes))

        branch = branches[branch_idx]

        # make drive for original node sequence

        for pulse_ctr, node in enumerate(branch):

            pulse_start = to_time_steps(pulse_ctr * INTER_PULSE_INTERVAL)
            pulse_end = pulse_start + to_time_steps(PULSE_DURATION)

            drives_partial[pulse_start:pulse_end, node] = PULSE_HEIGHT

        # make replay trigger pulse

        pulse_start = to_time_steps(REPLAY_PULSE_START)
        pulse_end = pulse_start + to_time_steps(PULSE_DURATION)

        drives_partial[pulse_start:pulse_end, branch[0]] = PULSE_HEIGHT

        # make reset pulse

        pulse_start = to_time_steps(RESET_PULSE_START)
        pulse_end = pulse_start + to_time_steps(RESET_PULSE_DURATION)

        drives_partial[pulse_start:pulse_end, :] = RESET_PULSE_HEIGHT

        drives.append(drives_partial)

    # convert to one long drive array

    drives = np.concatenate(drives, axis=0)

    # run network

    v_0s = V_REST * np.ones((n_nodes,))

    vs, rs = ntwk.run(v_0s, drives, DT)

    return n_nodes, drives, vs, rs


def tree_structure_replay_demo(
        SEED,
        TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        DRIVE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_PULSE_START,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT,
        FIG_SIZE, VERT_SPACING, FONT_SIZE):
    """
    Run demo for sequential replay in tree structured network.
    See _tree_structure_replay_demo_simulation for more details

    :param FIG_SIZE: figure size
    :param VERT_SPACING: spacing between neuron firing rate traces
    :param FONT_SIZE: font size
    """

    n_nodes, drives, vs, rs = _tree_structure_replay_demo_simulation(
        SEED, TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        DRIVE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_PULSE_START,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT)

    # plot things

    ts = np.arange(len(rs)) * DT

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    multivariate_same_axis(
        ax, ts=ts, data=[rs, drives], scales=[1, PULSE_HEIGHT], spacing=VERT_SPACING,
        colors=['k', 'r'], lw=2)

    ax.set_xlim(0, DRIVE_START + INTER_TRAIN_INTERVAL*len(BRANCH_ORDER))
    ax.set_ylim(-VERT_SPACING, n_nodes * VERT_SPACING)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('node')

    set_fontsize(ax, FONT_SIZE)


def tree_structure_replay_demo_heat_map(
        SEED,
        TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        DRIVE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_PULSE_START,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT,
        FIG_SIZE, V_MIN, V_MAX, FONT_SIZE):
    """
    Run demo for sequential replay in tree structured network.
    See _tree_structure_replay_demo_simulation for more details

    :param FIG_SIZE: figure size
    :param VERT_SPACING: spacing between neuron firing rate traces
    :param FONT_SIZE: font size
    """

    n_nodes, drives, vs, rs = _tree_structure_replay_demo_simulation(
        SEED, TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        DRIVE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_PULSE_START,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT)

    # plot things

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    firing_rate_heat_map(ax, DT, rs, vmin=V_MIN, vmax=V_MAX)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('node')

    set_fontsize(ax, FONT_SIZE)

    return fig


def chain_propagation_demo(
        N_PRINCIPAL_NODES, FIG_SIZE, FONT_SIZE):
    """
    Demo propagation of activity through a rate-based model.

    :param N_NODES: number of nodes
    :param FIG_SIZE: figure size
    :param FONT_SIZE: font size
    """

    # choose parameters that we know work for successful propagation

    SEED = 0

    TAUS = 0.03 * np.ones((2 * N_PRINCIPAL_NODES,))
    V_RESTS = -0.06 * np.ones((2 * N_PRINCIPAL_NODES,))
    V_THS = -0.02 * np.ones((2 * N_PRINCIPAL_NODES,))
    GAINS = 100 * np.ones((2 * N_PRINCIPAL_NODES,))
    NOISES = .060 * np.ones((2 * N_PRINCIPAL_NODES,))

    W_PP = 0.08
    W_MP = 0
    W_PM = 0
    W_MM = 0

    PULSE_START = 0.2
    PULSE_HEIGHT = 1.5
    PULSE_END = 0.25

    SIM_DURATION = 3
    DT = 0.001

    VERT_SPACING = 1

    np.random.seed(SEED)

    # make connectivity matrix

    principal_mask = np.zeros(
        (N_PRINCIPAL_NODES, N_PRINCIPAL_NODES), dtype=float)

    for node in range(N_PRINCIPAL_NODES - 1):

        principal_mask[node + 1, node] = 1

    w = connectivity.basic_adlib(
        principal_connectivity_mask=principal_mask,
        w_pp=W_PP, w_mp=W_MP, w_pm=W_PM, w_mm=W_MM)

    # make network

    ntwk = network.RateBasedModel(
        taus=TAUS, v_rests=V_RESTS, v_ths=V_THS, gains=GAINS, noises=NOISES,
        w=w)

    # make pulse input

    n_time_steps = int(SIM_DURATION // DT)

    drives = np.zeros((n_time_steps, 2 * N_PRINCIPAL_NODES), dtype=float)
    drives[int(PULSE_START // DT):int(PULSE_END // DT), 0] = PULSE_HEIGHT

    v_0s = V_RESTS.copy()

    vs, rs = ntwk.run(v_0s, drives, dt=DT)

    # make figure

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    # plot input

    ts = np.arange(n_time_steps) * DT

    ax.plot(ts, (drives[:, 0] > 0).astype(float) + VERT_SPACING, c='g', lw=2)

    for node in range(N_PRINCIPAL_NODES):

        # plot firing rates

        y_offset = -node * VERT_SPACING

        ax.plot(ts, rs[:, node] + y_offset, c='k', lw=2)
        ax.axhline(y_offset, color='gray', ls='--')

    ax.set_yticks(-np.arange(-1, N_PRINCIPAL_NODES) * VERT_SPACING)
    ax.set_yticklabels(['input'] + range(N_PRINCIPAL_NODES))

    ax.set_xlabel('time (s)')
    ax.set_ylabel('node')

    set_fontsize(ax, FONT_SIZE)


def self_excitation_bistability(
        VS, TAU, V_REST, V_TH, GAIN, W_SELFS, FIG_SIZE, FONT_SIZE):
    """
    Plot dv/dt vs. v for a rate-based population with several self-excitation
    strengths.
    """

    dv_dts = []

    for w_self in W_SELFS:

        dv_dt = (1 / TAU) * (-(VS - V_REST) + w_self * sigmoid(GAIN * (VS - V_TH)))
        dv_dts.append(dv_dt)

    _, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    lines = []
    for w_self, dv_dt in zip(W_SELFS, dv_dts):

        line, = ax.plot(VS, dv_dt, lw=3, label='W_self = {}'.format(w_self))
        lines.append(line)

    ax.axhline(0, color='gray', lw=2, ls='--')
    ax.set_xlabel('v (V)')
    ax.set_ylabel('dv/dt (V/s)')

    ax.legend(handles=lines, loc='best')

    set_fontsize(ax, FONT_SIZE)


def threshold_potential_analysis(
        VS, V_REST, V_TH, GAIN, W_PM, W_PP, W_TOGGLE,
        FIG_SIZE, FONT_SIZE):
    """
    Plot the threshold plus the various "steady" states caused by
    different levels of input to a node.
    """

    fig, axs = plt.subplots(
        2, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    rs = sigmoid(GAIN * (VS - V_TH))

    axs[0].plot(VS, rs, lw=3, color='b')
    axs[0].axvline(V_REST, lw=3, color='gray', ls='--')
    axs[0].axvline(V_REST + W_PM, lw=3, color='gray', ls='-')
    axs[0].axvline(V_REST + W_PP, lw=3, color='k', ls='--')
    axs[0].axvline(V_REST + W_PP + W_PM, lw=3, color='k', ls='-')

    axs[1].plot(VS, rs, lw=3, color='b')
    axs[1].axvline(V_REST + W_TOGGLE, lw=3, color='gray', ls='--')
    axs[1].axvline(V_REST + W_PM + W_TOGGLE, lw=3, color='gray', ls='-')
    axs[1].axvline(V_REST + W_PP + W_TOGGLE, lw=3, color='k', ls='--')
    axs[1].axvline(V_REST + W_PP + W_PM + W_TOGGLE, lw=3, color='k', ls='-')

    axs[0].set_ylabel('Firing rate')
    axs[1].set_xlabel('Voltage (V)')
    axs[1].set_ylabel('Firing rate')

    axs[0].set_title('stimulus/memory driven')
    axs[1].set_title('spontaneous')

    axs[0].legend([
        'firing rate',
        'V_REST',
        'V_REST + W_PM',
        'V_REST + W_PP',
        'V_REST + W_PM + W_PP'
    ], loc='best')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)


def single_ensemble_hysteresis_demo(
        SEED,
        TAU, V_REST, V_TH, GAIN, NOISE, DT, W_MP, W_PM, W_MM,
        PULSE_HEIGHT_SMALL, PULSE_HEIGHT_LARGE, PULSE_DURATION, INTER_PULSE_INTERVAL,
        SIM_DURATION, W_MMS_PHASE_PLOT,
        FIG_SIZE, VERTICAL_SPACING, FONT_SIZE):

    # make network

    w = np.array([
        [   0, W_PM],
        [W_MP, W_MM],
    ])

    ntwk = network.RateBasedModel(
        taus=TAU*np.ones((2,)),
        v_rests=V_REST*np.ones((2,)),
        v_ths=V_TH*np.ones((2,)),
        gains=GAIN*np.ones((2,)),
        noises=NOISE*np.ones((2,)),
        w=w,
    )

    # make drive

    n_steps = SIM_DURATION // DT

    drives = np.zeros((n_steps, 2))

    intvl = int(INTER_PULSE_INTERVAL // DT)
    dur = int(PULSE_DURATION // DT)

    # first small pulse

    start = intvl

    drives[start:start + dur, 0] = PULSE_HEIGHT_SMALL

    # large pulse

    start = 2 * intvl

    drives[start:start + dur, 0] = PULSE_HEIGHT_LARGE

    # second small pulse

    start = 3*intvl

    drives[start:start + dur, 0] = PULSE_HEIGHT_SMALL

    # run simulation

    v_0s = V_REST * np.ones((2,))

    np.random.seed(SEED)

    vs, rs = ntwk.run(v_0s, drives, DT)

    ts = np.arange(n_steps) * DT

    # make phase plots for the memory units

    vs_memory = np.linspace(-.1, 0.06, 1000)

    dv_dts = []

    for w_mm in W_MMS_PHASE_PLOT:

        dv_dt = (1 / TAU) * (-(vs_memory - V_REST) + \
            w_mm * sigmoid(GAIN * (vs_memory - V_TH)))

        dv_dts.append(dv_dt)


    # MAKE PLOTS

    fig = plt.figure(facecolor='white', figsize=FIG_SIZE, tight_layout=True)

    gs = gridspec.GridSpec(1, 3)

    ax = fig.add_subplot(gs[:2])

    # plot stimulus

    drive_plot = 2 * drives[:, 0] / PULSE_HEIGHT_LARGE
    ax.plot(ts, drive_plot, c='r', lw=2)
    ax.axhline(2, c='r', lw=2, ls='--')

    # plot voltages

    v_plot_0 = vs[:, 0] - V_REST
    v_plot_0 /= (V_TH - V_REST)
    v_plot_0 += 2 * VERTICAL_SPACING

    ax.plot(ts, v_plot_0, c='k', lw=2)
    ax.axhline(2 * VERTICAL_SPACING + 1, color='gray', lw=2, ls='--')

    v_plot_1 = vs[:, 1] - V_REST
    v_plot_1 /= (V_TH - V_REST)
    v_plot_1 += VERTICAL_SPACING

    ax.plot(ts, v_plot_1, c='k', lw=2)
    ax.axhline(VERTICAL_SPACING + 1, color='gray', lw=2, ls='--')

    ax.set_yticks([
        0, 2, VERTICAL_SPACING, VERTICAL_SPACING + 1,
        2 * VERTICAL_SPACING, 2 * VERTICAL_SPACING + 1,
    ])

    ax.set_yticklabels(
        [0, r'$s_{max}$', r'$v_{rest}$', r'$v_{th}$', r'$v_{rest}$', r'$v_{th}$']
    )

    ax.get_yticklabels()[1].set_color('r')

    ax.text(0.01, 1.2, 'stimulus', color='r', fontsize=FONT_SIZE)
    ax.text(0.01, VERTICAL_SPACING + 1.3, 'memory unit', fontsize=FONT_SIZE)
    ax.text(0.01, 2 * VERTICAL_SPACING + 1.3, 'primary unit', fontsize=FONT_SIZE)

    ax.set_xlim(0, SIM_DURATION)
    ax.set_ylim(0, 3.1 * VERTICAL_SPACING)

    ax.set_xlabel('time (s)', color='k')
    ax.set_ylabel('stimulus, voltage', color='k')

    # plot firing rates

    ax_twin = ax.twinx()

    r_plot_0 = 1.5 * rs[:, 0] + 2 * VERTICAL_SPACING

    ax_twin.plot(ts, r_plot_0, c='g', lw=2)

    r_plot_1 = 1.5 * rs[:, 1] + VERTICAL_SPACING

    ax_twin.plot(ts, r_plot_1, c='g', lw=2)

    ax_twin.set_yticks([
        VERTICAL_SPACING, VERTICAL_SPACING + 1.5,
        2 * VERTICAL_SPACING, 2 * VERTICAL_SPACING + 1.5,
    ])

    ax_twin.set_yticklabels([0, 1, 0, 1])

    ax_twin.set_ylim(0, 3.5 * VERTICAL_SPACING)

    ax_twin.set_ylabel('firing rate', color='g')
    [tl.set_color('g') for tl in ax_twin.get_yticklabels()]

    ax_2 = fig.add_subplot(gs[2])

    for dv_dt in dv_dts:

        ax_2.plot(vs_memory, dv_dt, lw=4)

    ax_2.axhline(0, color='gray', lw=3, ls='--')

    ax_2.set_xticks([-.1, -.02, .06])
    ax_2.yaxis.tick_right()

    ax_2.set_xlabel('voltage (V)')
    ax_2.set_ylabel('dv/dt (V/s)')

    ax_2.yaxis.set_label_position('right')

    ax_2.legend(W_MMS_PHASE_PLOT)

    for ax_temp in [ax, ax_twin, ax_2]:

        set_fontsize(ax_temp, FONT_SIZE)

    return fig


def tree_structure_demo_global_control(
        SEED, TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        DRIVE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_PULSE_START,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT,
        AMNESIA_PULSE_START, AMNESIA_PULSE_DURATION, AMNESIA_PULSE_HEIGHT,
        SPONTANEOUS_PULSE_START, SPONTANEOUS_PULSE_DURATION, SPONTANEOUS_PULSE_HEIGHT,
        SIM_DURATION, FIG_SIZE, V_MIN, V_MAX, FONT_SIZE):

    def to_time_steps(interval):
        return int(interval / DT)

    np.random.seed(SEED)

    w, branches = _build_tree_structure_demo_connectivity(
        branch_length=BRANCH_LENGTH,
        w_pp=W_PP, w_mp=W_MP, w_pm=W_PM, w_mm=W_MM)

    n_nodes = w.shape[0]

    # build final network

    ntwk = network.RateBasedModel(
        taus=TAU * np.ones((n_nodes,)),
        v_rests=V_REST * np.ones((n_nodes,)),
        v_ths=V_TH * np.ones((n_nodes,)),
        gains=GAIN * np.ones((n_nodes,)),
        noises=NOISE * np.ones((n_nodes,)),
        w=w)

    # build drive sequences

    drives = [np.zeros((to_time_steps(DRIVE_START), n_nodes))]

    for branch_idx in BRANCH_ORDER:

        drives_partial = np.zeros((to_time_steps(INTER_TRAIN_INTERVAL), n_nodes))

        branch = branches[branch_idx]

        # make drive for original node sequence

        for pulse_ctr, node in enumerate(branch):
            pulse_start = to_time_steps(pulse_ctr * INTER_PULSE_INTERVAL)
            pulse_end = pulse_start + to_time_steps(PULSE_DURATION)

            drives_partial[pulse_start:pulse_end, node] = PULSE_HEIGHT

        # make replay trigger pulse

        pulse_start = to_time_steps(REPLAY_PULSE_START)
        pulse_end = pulse_start + to_time_steps(PULSE_DURATION)

        drives_partial[pulse_start:pulse_end, branch[0]] = PULSE_HEIGHT

        # make reset pulse

        pulse_start = to_time_steps(RESET_PULSE_START)
        pulse_end = pulse_start + to_time_steps(RESET_PULSE_DURATION)

        drives_partial[pulse_start:pulse_end, :] = RESET_PULSE_HEIGHT

        drives.append(drives_partial)

    # convert to one long drive array

    drives_temp = np.concatenate(drives, axis=0)
    drives = np.zeros((to_time_steps(SIM_DURATION), n_nodes))
    drives[:len(drives_temp), :] = drives_temp

    # add in global signals

    # amnesia pulse (inhibitory input to memory units)

    start = to_time_steps(AMNESIA_PULSE_START)
    end = start + to_time_steps(AMNESIA_PULSE_DURATION)
    drives[start:end, n_nodes/2:] = AMNESIA_PULSE_HEIGHT

    # spontaneous pulse

    start = to_time_steps(SPONTANEOUS_PULSE_START)
    end = start + to_time_steps(SPONTANEOUS_PULSE_DURATION)
    drives[start:end, :n_nodes/2] = SPONTANEOUS_PULSE_HEIGHT

    # run network

    v_0s = V_REST * np.ones((n_nodes,))

    vs, rs = ntwk.run(v_0s, drives, DT)

    # plot things

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    firing_rate_heat_map(ax, DT, rs, vmin=V_MIN, vmax=V_MAX)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('node')

    set_fontsize(ax, FONT_SIZE)

    return fig