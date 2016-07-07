from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from connectivity import feed_forward_grid, er_directed

import metrics

from network import LIFExponentialSynapsesModel
from network import SoftmaxWTAWithLingeringHyperexcitability

from plot import fancy_raster, fancy_raster_arrows_above
from plot import multivariate_same_axis
from plot import set_fontsize


def lif_demo_two_branches(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M,
        TAUS_SYN, V_REVS_SYN,
        BRANCH_LEN,
        W_PP, W_PI, W_PM, W_IP, W_II, W_MP, W_MM,
        BKGD_STARTS, BKGD_ENDS, BKGD_FRQS, BKGD_STRENS,
        DRIVE_ORDERS, DRIVE_STARTS, DRIVE_STRENS, DRIVE_DURS, DRIVE_FRQS, DRIVE_ITVS,
        REPLAY_TRIGGER_TIMES, REPLAY_TRIGGER_STRENS,
        MEMORY_RESET_STARTS, MEMORY_RESET_ENDS, MEMORY_RESET_STRENS, MEMORY_RESET_FRQS,
        SIM_DURATION, DT,
        P_4_M_4_PLOT_LIMITS, P_3_P_4_P_7_PLOT_LIMITS,
        FIG_SIZE, FONT_SIZE):


    syns = TAUS_SYN.keys()

    # build drive arrays

    np.random.seed(SEED)

    n_steps = int(SIM_DURATION / DT)
    n_primary_cells = 3 * BRANCH_LEN
    n_cells = 2 * n_primary_cells + 1

    drives = {syn: np.zeros((n_steps, n_cells)) for syn in syns}

    for syn in syns:

        for drive_ctr in range(len(DRIVE_ORDERS)):

            # drive one path through network with a sequence of volleys

            start = DRIVE_STARTS[drive_ctr][syn]
            dur = DRIVE_DURS[drive_ctr][syn]
            frq = DRIVE_FRQS[drive_ctr][syn]
            itv = DRIVE_ITVS[drive_ctr][syn]

            if frq > 0:

                period = 1. / frq

            else:

                period = np.inf

            for cell_ctr, cell in enumerate(DRIVE_ORDERS[drive_ctr]):
                volley_start = start + cell_ctr * itv
                volley_end = volley_start + dur
                volley_times = np.arange(volley_start, volley_end, period)

                volley_times_idx = np.round(volley_times / DT).astype(int)

                drives[syn][volley_times_idx, cell] += DRIVE_STRENS[drive_ctr][syn]

            # drive first neuron with replay trigger

            replay_trigger_time = REPLAY_TRIGGER_TIMES[drive_ctr][syn]
            replay_trigger_time_idx = np.round(replay_trigger_time / DT).astype(int)

            drives[syn][replay_trigger_time_idx, DRIVE_ORDERS[drive_ctr][0]] += \
                REPLAY_TRIGGER_STRENS[drive_ctr][syn]

        for mr_ctr in range(len(MEMORY_RESET_STARTS)):

            # reset memory units

            mr_start = MEMORY_RESET_STARTS[mr_ctr][syn]
            mr_end = MEMORY_RESET_ENDS[mr_ctr][syn]
            mr_stren = MEMORY_RESET_STRENS[mr_ctr][syn]
            mr_frq = MEMORY_RESET_FRQS[mr_ctr][syn]

            if mr_frq > 0:

                mr_period = 1. / mr_frq

            else:

                mr_period = np.inf

            mr_times = np.arange(mr_start, mr_end, mr_period)
            mr_times_idx = np.round(mr_times / DT).astype(int)

            drives[syn][mr_times_idx, n_primary_cells:2 * n_primary_cells] += mr_stren

        # add background

        for bkgd_ctr in range(len(BKGD_STARTS)):
            bkgd_start_idx = int(BKGD_STARTS[bkgd_ctr][syn] / DT)
            bkgd_end_idx = int(BKGD_ENDS[bkgd_ctr][syn] / DT)
            bkgd_frq = BKGD_FRQS[bkgd_ctr][syn]
            bkgd_stren = BKGD_STRENS[bkgd_ctr][syn]

            bkgd = bkgd_stren * np.random.poisson(
                bkgd_frq * DT, (bkgd_end_idx - bkgd_start_idx, n_primary_cells))

            drives[syn][bkgd_start_idx:bkgd_end_idx, :n_primary_cells] += bkgd

    # build weight matrices

    ws = {syn: np.zeros((n_cells, n_cells)) for syn in syns}

    for syn in syns:
        # branching chain

        ws[syn][range(1, n_primary_cells), range(0, n_primary_cells - 1)] = W_PP[syn]

        temp_idx = 2 * BRANCH_LEN

        ws[syn][temp_idx, temp_idx - 1] = 0
        ws[syn][temp_idx, BRANCH_LEN - 1] = W_PP[syn]

        # to primary from inhibitory

        ws[syn][:n_primary_cells, -1] = W_PI[syn]

        # to primary from memory

        ws[syn][range(n_primary_cells), range(n_primary_cells, 2 * n_primary_cells)] = W_PM[syn]

        # to inhibitory from primary

        ws[syn][-1, :n_primary_cells] = W_IP[syn]

        # to inhibitory from inhibitory

        ws[syn][-1, -1] = W_II[syn]

        # to memory from primary

        ws[syn][range(n_primary_cells, 2 * n_primary_cells), range(n_primary_cells)] = W_MP[syn]

        # memory to memory

        ws[syn][range(n_primary_cells, 2 * n_primary_cells), range(n_primary_cells, 2 * n_primary_cells)] = W_MM[syn]

    # make network

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

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)

    # MAKE PLOTS

    fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, tight_layout=True)

    # primary spikes

    primary_spikes = measurements['spikes'][:, :n_primary_cells].nonzero()

    axs[0].scatter(primary_spikes[0] * DT, primary_spikes[1], s=200, marker='|', c='k', lw=1)

    axs[0].set_xlim(0, SIM_DURATION)
    axs[0].set_ylim(-1, n_primary_cells)

    axs[0].set_ylabel('neuron')
    axs[0].set_yticks([0, 2, 4, 6, 8])
    axs[0].set_yticklabels(['P1', 'P3', 'P5', 'P7', 'P9'])
    axs[0].set_title('primary neuron spikes')

    # memory spikes

    memory_spikes = measurements['spikes'][:, n_primary_cells:2 * n_primary_cells].nonzero()

    axs[1].scatter(memory_spikes[0] * DT, memory_spikes[1], s=150, marker='|', c='b', lw=1)

    axs[1].set_xlim(0, SIM_DURATION)
    axs[1].set_ylim(-1, n_primary_cells)

    axs[1].set_ylabel('neuron')
    axs[1].set_yticks([0, 2, 4, 6, 8])
    axs[1].set_yticklabels(['M1', 'M3', 'M5', 'M7', 'M9'])
    axs[1].set_title('memory neuron spikes')

    # example primary and memory voltages during initial drive

    p_4_voltage = measurements['voltages'][:, 3]
    m_4_voltage = measurements['voltages'][:, n_primary_cells + 3]

    ts = np.arange(n_steps + 1) * DT

    handles = []

    handles.append(axs[2].plot(ts, 1000 * p_4_voltage, color='k', lw=2, label='P4', zorder=1)[0])
    handles.append(axs[2].plot(ts, 1000 * m_4_voltage, color='b', lw=1, label='M4', zorder=0)[0])

    axs[2].axhline(1000 * V_TH, color='gray', ls='--')

    axs[2].set_xlim(*P_4_M_4_PLOT_LIMITS)
    axs[2].set_ylim(-90, 0)
    axs[2].set_ylabel('voltage (mV)')
    axs[2].set_title('voltages during initial stimulus')
    axs[2].legend(handles=handles)

    # example primary voltages during replay

    p_3_voltage = measurements['voltages'][:, 2]
    p_4_voltage = measurements['voltages'][:, 3]
    p_7_voltage = measurements['voltages'][:, 6]

    handles = []

    handles.append(axs[3].plot(ts, 1000 * p_3_voltage, color='r', lw=2, label='P3')[0])
    handles.append(axs[3].plot(ts, 1000 * p_4_voltage, color='k', lw=2, label='P4')[0])
    handles.append(axs[3].plot(ts, 1000 * p_7_voltage, color='g', lw=2, label='P7')[0])

    axs[3].axhline(1000 * V_TH, color='gray', ls='--')

    axs[3].set_xlim(*P_3_P_4_P_7_PLOT_LIMITS)
    axs[3].set_ylim(-90, 0)
    axs[3].set_ylabel('voltage (mV)')
    axs[3].set_title('voltages during replay')
    axs[3].legend(handles=handles)

    axs[-1].set_xlabel('time (s)')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def lif_demo_two_branches_global_controls(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M,
        TAUS_SYN, V_REVS_SYN,
        BRANCH_LEN,
        W_PP, W_PI, W_PM, W_IP, W_II, W_MP, W_MM,
        BKGD_STARTS, BKGD_ENDS, BKGD_FRQS, BKGD_STRENS,
        DRIVE_ORDERS, DRIVE_STARTS, DRIVE_STRENS, DRIVE_DURS, DRIVE_FRQS, DRIVE_ITVS,
        REPLAY_TRIGGER_TIMES, REPLAY_TRIGGER_STRENS,
        MEMORY_INH_STARTS, MEMORY_INH_ENDS, MEMORY_INH_STRENS, MEMORY_INH_FRQS,
        SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE):

    syns = TAUS_SYN.keys()

    # build drive arrays

    np.random.seed(SEED)

    n_steps = int(SIM_DURATION / DT)
    n_primary_cells = 3 * BRANCH_LEN
    n_cells = 2 * n_primary_cells + 1

    drives = {syn: np.zeros((n_steps, n_cells)) for syn in syns}

    for syn in syns:

        for drive_ctr in range(len(DRIVE_ORDERS)):

            # drive one path through network with a sequence of volleys

            start = DRIVE_STARTS[drive_ctr][syn]
            dur = DRIVE_DURS[drive_ctr][syn]
            frq = DRIVE_FRQS[drive_ctr][syn]
            itv = DRIVE_ITVS[drive_ctr][syn]

            if frq > 0:

                period = 1. / frq

            else:

                period = np.inf

            for cell_ctr, cell in enumerate(DRIVE_ORDERS[drive_ctr]):
                volley_start = start + cell_ctr * itv
                volley_end = volley_start + dur
                volley_times = np.arange(volley_start, volley_end, period)

                volley_times_idx = np.round(volley_times / DT).astype(int)

                drives[syn][volley_times_idx, cell] += DRIVE_STRENS[drive_ctr][syn]

            # drive first neuron with replay trigger

            replay_trigger_time = REPLAY_TRIGGER_TIMES[drive_ctr][syn]
            replay_trigger_time_idx = np.round(replay_trigger_time / DT).astype(int)

            drives[syn][replay_trigger_time_idx, DRIVE_ORDERS[drive_ctr][0]] += \
                REPLAY_TRIGGER_STRENS[drive_ctr][syn]

        for mr_ctr in range(len(MEMORY_INH_STARTS)):

            # reset memory units

            mr_start = MEMORY_INH_STARTS[mr_ctr][syn]
            mr_end = MEMORY_INH_ENDS[mr_ctr][syn]
            mr_stren = MEMORY_INH_STRENS[mr_ctr][syn]
            mr_frq = MEMORY_INH_FRQS[mr_ctr][syn]

            if mr_frq > 0:

                mr_period = 1. / mr_frq

            else:

                mr_period = np.inf

            mr_times = np.arange(mr_start, mr_end, mr_period)
            mr_times_idx = np.round(mr_times / DT).astype(int)

            drives[syn][mr_times_idx, n_primary_cells:2 * n_primary_cells] += mr_stren

        # add background

        for bkgd_ctr in range(len(BKGD_STARTS)):
            bkgd_start_idx = int(BKGD_STARTS[bkgd_ctr][syn] / DT)
            bkgd_end_idx = int(BKGD_ENDS[bkgd_ctr][syn] / DT)
            bkgd_frq = BKGD_FRQS[bkgd_ctr][syn]
            bkgd_stren = BKGD_STRENS[bkgd_ctr][syn]

            bkgd = bkgd_stren * np.random.poisson(
                bkgd_frq * DT, (bkgd_end_idx - bkgd_start_idx, n_primary_cells))

            drives[syn][bkgd_start_idx:bkgd_end_idx, :n_primary_cells] += bkgd

    # build weight matrices

    ws = {syn: np.zeros((n_cells, n_cells)) for syn in syns}

    for syn in syns:
        # branching chain

        ws[syn][range(1, n_primary_cells), range(0, n_primary_cells - 1)] = W_PP[syn]

        temp_idx = 2 * BRANCH_LEN

        ws[syn][temp_idx, temp_idx - 1] = 0
        ws[syn][temp_idx, BRANCH_LEN - 1] = W_PP[syn]

        # to primary from inhibitory

        ws[syn][:n_primary_cells, -1] = W_PI[syn]

        # to primary from memory

        ws[syn][range(n_primary_cells), range(n_primary_cells, 2 * n_primary_cells)] = W_PM[syn]

        # to inhibitory from primary

        ws[syn][-1, :n_primary_cells] = W_IP[syn]

        # to inhibitory from inhibitory

        ws[syn][-1, -1] = W_II[syn]

        # to memory from primary

        ws[syn][range(n_primary_cells, 2 * n_primary_cells), range(n_primary_cells)] = W_MP[syn]

        # memory to memory

        ws[syn][range(n_primary_cells, 2 * n_primary_cells), range(n_primary_cells, 2 * n_primary_cells)] = W_MM[syn]

    # make network

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

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)

    # MAKE PLOTS

    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, tight_layout=True)

    # primary spikes

    primary_spikes = measurements['spikes'][:, :n_primary_cells].nonzero()

    axs[0].scatter(primary_spikes[0] * DT, primary_spikes[1], s=200, marker='|', c='k', lw=1)

    axs[0].set_xlim(0, SIM_DURATION)
    axs[0].set_ylim(-1, n_primary_cells)

    axs[0].set_ylabel('neuron')
    axs[0].set_yticks([0, 2, 4, 6, 8])
    axs[0].set_yticklabels(['P1', 'P3', 'P5', 'P7', 'P9'])
    axs[0].set_title('primary neuron spikes')

    # memory spikes

    memory_spikes = measurements['spikes'][:, n_primary_cells:2 * n_primary_cells].nonzero()

    axs[1].scatter(memory_spikes[0] * DT, memory_spikes[1], s=150, marker='|', c='b', lw=1)

    axs[1].set_xlim(0, SIM_DURATION)
    axs[1].set_ylim(-1, n_primary_cells)

    axs[1].set_ylabel('neuron')
    axs[1].set_yticks([0, 2, 4, 6, 8])
    axs[1].set_yticklabels(['M1', 'M3', 'M5', 'M7', 'M9'])
    axs[1].set_title('memory neuron spikes')

    axs[-1].set_xlabel('time (s)')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)

    return fig


def simplified_ff_replay_example(
        SEED, GRID_SHAPE, LATERAL_SPREAD, G_W, G_X, G_D, T_X,
        DRIVEN_NODES, DRIVE_AMPLITUDE, SPONTANEOUS_RUN_TIME,
        AX_SIZE, Y_LIM, FONT_SIZE):
    """
    Show a few examples of how a basic network with activation-dependent lingering hyperexcitability
    with a feed forward architecture can replay certain sequences.
    """

    def make_feed_forward_grid_weights(shape, spread):
        """
        Create a weight matrix corresponding to a network of feed-forward layers, with lateral
        spread from each node over nodes in the successive layer.
        :param shape: grid shape of the network (n_rows, n_cols)
        :param spread: size of lateral spread (1 = no lateral spread)
        :return: weight matrix (rows are targs, cols are srcs)
        """

        w = feed_forward_grid(shape=shape, spread=1)
        w += feed_forward_grid(shape=shape, spread=spread)

        w = (w > 0).astype(float)

        return w

    # RUN SIMULATION

    np.random.seed(SEED)
    n_trials = len(DRIVEN_NODES)

    # make network

    w = make_feed_forward_grid_weights(GRID_SHAPE, LATERAL_SPREAD)
    ntwk = SoftmaxWTAWithLingeringHyperexcitability(w=w, g_w=G_W, g_x=G_X, g_d=G_D, t_x=T_X)

    n_nodes = w.shape[0]

    # loop over trials

    all_drives = []
    all_rs = []

    for driven_nodes_one_trial in DRIVEN_NODES:

        run_time = 2 * len(driven_nodes_one_trial)

        # make drive sequence

        drives = np.zeros((run_time, n_nodes), dtype=float)

        for t, node in enumerate(driven_nodes_one_trial):

            drives[t, node] = DRIVE_AMPLITUDE

        drives[len(driven_nodes_one_trial), driven_nodes_one_trial[0]] = DRIVE_AMPLITUDE

        # run network

        r_0 = np.zeros((n_nodes,))
        xc_0 = np.zeros((n_nodes,))
        rs = ntwk.run(r_0=r_0, drives=drives, xc_0=xc_0)

        all_drives.append(drives)
        all_rs.append(rs)

    # make spontaneous example

    drives_spontaneous = np.zeros((SPONTANEOUS_RUN_TIME, n_nodes), dtype=float)
    nonzero_drives_spontaneous = all_drives[0][:len(driven_nodes_one_trial)]

    drives_spontaneous[:len(nonzero_drives_spontaneous)] = nonzero_drives_spontaneous

    # run network

    r_0_spontaneous = np.zeros((n_nodes,))
    xc_0_spontaneous = np.zeros((n_nodes,))
    rs_spontaneous = ntwk.run(r_0=r_0_spontaneous, drives=drives_spontaneous, xc_0=xc_0_spontaneous)

    # MAKE PLOTS

    fig = plt.figure(
        figsize=(AX_SIZE[0] * (n_trials + 2), AX_SIZE[1]), facecolor='white',
        tight_layout=True)
    axs = []

    axs.append(fig.add_subplot(1, 4, 1))
    axs.append(fig.add_subplot(1, 4, 2, sharey=axs[0]))

    axs.append(fig.add_subplot(1, 2, 2, sharey=axs[0]))

    for ax, drives, rs in zip(axs[:-1], all_drives, all_rs):

        fancy_raster_arrows_above(ax, rs, drives, spike_marker_size=40, arrow_marker_size=80, rise=6)

        x_fill = np.linspace(-1, np.sum(drives > 0) - 1.5, 3, endpoint=True)
        y_fill_lower = -1 * np.ones(x_fill.shape)
        y_fill_upper = n_nodes * np.ones(x_fill.shape)

        ax.fill_between(x_fill, y_fill_lower, y_fill_upper, color='red', alpha=0.1)

    fancy_raster_arrows_above(
        axs[-1], rs_spontaneous, drives_spontaneous,
        spike_marker_size=40, arrow_marker_size=80, rise=6)

    x_fill = np.linspace(-1, np.sum(all_drives[0] > 0) - 1.5, 3, endpoint=True)
    y_fill_lower = -1 * np.ones(x_fill.shape)
    y_fill_upper = n_nodes * np.ones(x_fill.shape)

    axs[-1].fill_between(x_fill, y_fill_lower, y_fill_upper, color='red', alpha=0.1)

    for ax_ctr, ax in enumerate(axs):

        if ax_ctr < len(axs) - 1:

            ax.set_xlim(-1, 2 * len(DRIVEN_NODES[0]))
            ax.set_xticks([0, 4, 8, 12])

        else:

            ax.set_xlim(-1, SPONTANEOUS_RUN_TIME)

        ax.set_ylim(Y_LIM)

        ax.set_xlabel('time step')

        if ax_ctr == 0:

            ax.set_ylabel('ensemble')

        if ax_ctr < len(axs) - 1:

            ax.set_title('Triggered replay {}'.format(ax_ctr + 1))

        else:

            ax.set_title('Spontaneous replay')

        set_fontsize(ax, FONT_SIZE)

    return fig


def simplified_ff_properties(
        SEED, GRID_SHAPE, LATERAL_SPREAD, G_W, G_X, G_D, T_X,
        DRIVE_0, DRIVE_1A, DRIVE_1B, N_REPEATS,
        FOCUSED_STIM_0, FOCUSED_STIM_1, DISTRIBUTED_STIM,
        Y_MAX_DISTRIBUTED_STIM,
        FIG_SIZE, FONT_SIZE):
    """
    Demonstrate some properties of the simplified feed-forward network.

    1. Multiple sequences can be stored if they don't overlap.
    2. When they overlap the replay will be mixed and matched.
    3. Previous activation patterns and the hyperexcitability they elicit can bias network
    response to unfocused stimuli.
    """

    def make_feed_forward_grid_weights(shape, spread):
        """
        Create a weight matrix corresponding to a network of feed-forward layers, with lateral
        spread from each node over nodes in the successive layer.
        :param shape: grid shape of the network (n_rows, n_cols)
        :param spread: size of lateral spread (1 = no lateral spread)
        :return: weight matrix (rows are targs, cols are srcs)
        """

        w = feed_forward_grid(shape=shape, spread=1)
        w += feed_forward_grid(shape=shape, spread=spread)

        w = (w > 0).astype(float)

        return w

    np.random.seed(SEED)

    w = make_feed_forward_grid_weights(GRID_SHAPE, LATERAL_SPREAD)
    ntwk = SoftmaxWTAWithLingeringHyperexcitability(w=w, g_w=G_W, g_x=G_X, g_d=G_D, t_x=T_X)

    n_nodes = w.shape[0]

    # first demonstration

    # build drive array

    trigger_0 = [DRIVE_0[0]] + (len(DRIVE_0) - 1) * [None]
    trigger_1A = [DRIVE_1A[0]] + (len(DRIVE_1A) - 1) * [None]
    trigger_seq = N_REPEATS * (trigger_0 + trigger_1A)

    drive_seq_non_ovlp = np.concatenate([DRIVE_0, DRIVE_1A, trigger_seq])

    drives_non_ovlp = np.zeros((len(drive_seq_non_ovlp), n_nodes))

    for t, node in enumerate(drive_seq_non_ovlp):

        if node is not None:

            drives_non_ovlp[t, node] = 1

    # run network

    r_0 = np.zeros((n_nodes,))
    xc_0 = np.zeros((n_nodes,))
    rs_non_ovlp = ntwk.run(r_0=r_0, drives=drives_non_ovlp, xc_0=xc_0)

    # second demonstration

    # build drive array

    trigger_0 = [DRIVE_0[0]] + (len(DRIVE_0) - 1) * [None]
    trigger_1B = [DRIVE_1B[0]] + (len(DRIVE_1B) - 1) * [None]
    trigger_seq = N_REPEATS * (trigger_0 + trigger_1B)

    drive_seq_ovlp = np.concatenate([DRIVE_0, DRIVE_1B, trigger_seq])

    drives_ovlp = np.zeros((len(drive_seq_ovlp), n_nodes))

    for t, node in enumerate(drive_seq_ovlp):

        if node is not None:

            drives_ovlp[t, node] = 1

    # run network

    r_0 = np.zeros((n_nodes,))
    xc_0 = np.zeros((n_nodes,))
    rs_ovlp = ntwk.run(r_0=r_0, drives=drives_ovlp, xc_0=xc_0)

    # third demonstration

    run_time = len(FOCUSED_STIM_0) + len(DISTRIBUTED_STIM)

    # when FOCUSED_STIM_0 preceds the distributed stim

    drives_distributed_stim_0 = np.zeros((run_time, n_nodes))

    for t, node in enumerate(FOCUSED_STIM_0):

        drives_distributed_stim_0[t, node] = 1

    for t, node_pair in enumerate(DISTRIBUTED_STIM):

        drives_distributed_stim_0[t + len(FOCUSED_STIM_0), node_pair[0]] = 1
        drives_distributed_stim_0[t + len(FOCUSED_STIM_0), node_pair[1]] = 1

    # run network

    r_0 = np.zeros((n_nodes,))
    xc_0 = np.zeros((n_nodes,))
    rs_distributed_stim_0 = ntwk.run(r_0=r_0, drives=drives_distributed_stim_0, xc_0=xc_0)

    # when FOCUSED_STIM_1 precedes the distributed stim

    drives_distributed_stim_1 = np.zeros((run_time, n_nodes))

    for t, node in enumerate(FOCUSED_STIM_1):
        drives_distributed_stim_1[t, node] = 1

    for t, node_pair in enumerate(DISTRIBUTED_STIM):
        drives_distributed_stim_1[t + len(FOCUSED_STIM_1), node_pair[0]] = 1
        drives_distributed_stim_1[t + len(FOCUSED_STIM_1), node_pair[1]] = 1

    # run network

    r_0 = np.zeros((n_nodes,))
    xc_0 = np.zeros((n_nodes,))
    rs_distributed_stim_1 = ntwk.run(r_0=r_0, drives=drives_distributed_stim_1, xc_0=xc_0)

    # MAKE PLOTS

    fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, tight_layout=True)

    fancy_raster_arrows_above(
        axs[0], rs_non_ovlp, drives_non_ovlp,
        spike_marker_size=40, arrow_marker_size=80, rise=6)

    x_fill = [-0.5, 2 * len(DRIVE_0) - 0.5]
    y_fill_lower = [-1, -1]
    y_fill_upper = [n_nodes, n_nodes]

    axs[0].fill_between(x_fill, y_fill_lower, y_fill_upper, color='red', alpha=0.1)

    axs[0].set_xlim(-0.5, len(drive_seq_non_ovlp))
    axs[0].set_ylim(-1, n_nodes)
    axs[0].set_xlabel('time step')
    axs[0].set_ylabel('ensemble')
    axs[0].set_title('non-overlapping stim sequences')

    fancy_raster_arrows_above(
        axs[1], rs_ovlp, drives_ovlp,
        spike_marker_size=40, arrow_marker_size=80, rise=6)

    axs[1].fill_between(x_fill, y_fill_lower, y_fill_upper, color='red', alpha=0.1)

    axs[1].set_xlim(-0.5, len(drive_seq_ovlp))
    axs[1].set_ylim(-1, n_nodes)
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('ensemble')
    axs[1].set_title('overlapping stim sequences')

    fancy_raster_arrows_above(
        axs[2], rs_distributed_stim_0, drives_distributed_stim_0,
        spike_marker_size=40, arrow_marker_size=80, rise=2)

    x_fill = [-0.5, len(FOCUSED_STIM_0) - 0.5]
    y_fill_lower = [-1, -1]
    y_fill_upper = [n_nodes, n_nodes]

    axs[2].fill_between(x_fill, y_fill_lower, y_fill_upper, color='red', alpha=0.1)

    axs[2].set_xlim(-0.5, len(drives_distributed_stim_0))
    axs[2].set_ylim(-1, Y_MAX_DISTRIBUTED_STIM)
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('ensemble')
    axs[2].set_title('distributed stimulus response following sequence 1')

    fancy_raster_arrows_above(
        axs[3], rs_distributed_stim_1, drives_distributed_stim_1,
        spike_marker_size=40, arrow_marker_size=80, rise=2)

    axs[3].fill_between(x_fill, y_fill_lower, y_fill_upper, color='red', alpha=0.1)

    axs[3].set_xlim(-0.5, len(drives_distributed_stim_1))
    axs[3].set_ylim(-1, Y_MAX_DISTRIBUTED_STIM)
    axs[3].set_xlabel('time step')
    axs[3].set_ylabel('ensemble')
    axs[3].set_title('distributed stimulus response following sequence 2')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig


def simplified_connectivity_dependence(
        SEED,
        N_NODES, P_CONNECT,
        G_D, G_W, G_X, T_X,
        MATCH_PROPORTIONS,
        N_TRIALS, SEQ_LENGTHS,
        DENSITIES, SEQ_LENGTH_DENSITIES, NS,
        FIG_SIZE, FONT_SIZE, COLORS):
    """
    Show a set of plots demonstrating how the connectivity structure in the simplified
    model affects the model's ability to decode past and current stimuli.
    """

    # plots B and C: past decoding accuracy vs match proportion when mixed with
    # random and zero connectivity

    np.random.seed(SEED)

    controls = ['random', 'zero']

    # the following are indexed by [seq_len][trial][match_proportion_idx]

    decoding_accuracies_random = [{
       seq_len: [[] for _ in range(N_TRIALS)]
       for seq_len in SEQ_LENGTHS
       } for _ in range(2)]

    decoding_accuracies_zero = [{
      seq_len: [[] for _ in range(N_TRIALS)]
      for seq_len in SEQ_LENGTHS
      } for _ in range(2)]

    for control in controls:

        ## RUN SIMULATIONS AND RECORD DECODING ACCURACIES

        r_0 = np.zeros((N_NODES,))
        xc_0 = np.zeros((N_NODES,))

        for g_x_ctr, g_x in enumerate([G_X, 0]):

            for tr_ctr in range(N_TRIALS):

                # build original weight matrix and convert to drive transition probability distribution

                w_matched = er_directed(N_NODES, P_CONNECT)
                p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(w_matched, G_W)

                # build template random or zero matrices

                if control == 'random':

                    w_control = er_directed(N_NODES, P_CONNECT)

                elif control == 'zero':

                    w_control = er_directed(N_NODES, 0)

                for mp_ctr, match_proportion in enumerate(MATCH_PROPORTIONS):

                    # make mixed weight matrices

                    control_mask = np.random.rand(*w_matched.shape) < match_proportion

                    w = w_control.copy()
                    w[control_mask] = w_matched[control_mask]

                    # make network

                    ntwk = SoftmaxWTAWithLingeringHyperexcitability(
                        w, g_d=G_D, g_w=G_W, g_x=g_x, t_x=T_X)

                    # create random drive sequences

                    for seq_len in SEQ_LENGTHS:

                        drives = np.zeros((2 * seq_len, N_NODES))

                        drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
                        drives[0, drive_first] = 1

                        for ctr in range(seq_len - 1):
                            drive_last = np.argmax(drives[ctr])
                            drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

                            drives[ctr + 1, drive_next] = 1

                        # add trigger

                        drives[seq_len, drive_first] = 1

                        drive_seq = np.argmax(drives[:seq_len], axis=1)

                        rs = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives)

                        rs_seq = rs[seq_len:].argmax(axis=1)

                        # calculate decoding accuracy

                        acc = metrics.levenshtein(drive_seq, rs_seq)

                        if control == 'random':

                            decoding_accuracies_random[g_x_ctr][seq_len][tr_ctr].append(acc)

                        elif control == 'zero':

                            decoding_accuracies_zero[g_x_ctr][seq_len][tr_ctr].append(acc)

    # plot D: dependence on sparsity

    qs = DENSITIES
    l = SEQ_LENGTH_DENSITIES

    guaranteed_replay_probability = (1 - qs) ** ((l - 1)*(l - 2))

    expected_paths = np.zeros((len(NS), len(qs)))

    for n_ctr, n in enumerate(NS):

        factor = np.prod(n - np.arange(1, l))

        expected_paths[n_ctr, :] = factor * (qs ** (l - 1))


    ## MAKE PLOT

    fig, axs = plt.subplots(3, 2, figsize=FIG_SIZE, tight_layout=True)

    # B & C

    for c_ctr, control in enumerate(controls):

        for g_x_ctr, (g_x, ax) in enumerate(zip([G_X, 0], axs[c_ctr, :])):

            handles = []

            for seq_len, color in zip(SEQ_LENGTHS, COLORS):

                if control == 'random':

                    trials = np.array(decoding_accuracies_random[g_x_ctr][seq_len])

                elif control == 'zero':

                    trials = np.array(decoding_accuracies_zero[g_x_ctr][seq_len])

                acc_mean = np.mean(trials, axis=0)
                acc_sem = stats.sem(trials, axis=0)

                label = 'L = {}'.format(seq_len)

                handles.append(ax.plot(MATCH_PROPORTIONS, acc_mean, color=color, lw=3, label=label)[0])

                ax.fill_between(
                    MATCH_PROPORTIONS, acc_mean - acc_sem, acc_mean + acc_sem, color=color, alpha=.3)

            ax.set_xlabel('match proportion\n(mixed with {} connectivity)'.format(control))

            if g_x_ctr == 0:

                ax.set_title('Hyperexcitability on')

            elif g_x_ctr == 1:

                ax.set_title('Hyperexcitability off')

        axs[c_ctr, 0].set_ylabel('edit distance')

    # D

    axs[2, 0].semilogx(DENSITIES, guaranteed_replay_probability, color='k', lw=2)

    axs[2, 0].set_xlabel('q')
    axs[2, 0].set_ylabel('p(guaranteed replay)')

    ax_twin = axs[2, 0].twinx()

    handles = []

    for n_ctr, n in enumerate(NS):

        handles.append(
            ax_twin.loglog(DENSITIES, expected_paths[n_ctr], lw=2, label='N = {}'.format(n))[0])

    ax_twin.legend(handles=handles)

    ax_twin.set_ylabel('expected paths')

    for ax in list(axs.flat) + [ax_twin]:

        set_fontsize(ax, FONT_SIZE)

    return fig


def simplified_connectivity_dependence_current_stim_decoding(
        SEED,
        N_NODES, P_CONNECT, G_W, G_DS, G_D_EXAMPLE,
        N_TIME_POINTS, N_TIME_POINTS_EXAMPLE, DECODING_SEQUENCE_LENGTHS,
        MATCH_PROPORTIONS, N_TRIALS, MATCH_PROPORTION_SEQUENCE_LENGTHS,
        FIG_SIZE, COLORS, MATCH_PROPORTION_COLORS, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities. (when weight matrix is nonbinary)
    """

    np.random.seed(SEED)

    ## RUN FIRST SIMULATION -- VARYING STIMULUS INFLUENCE

    keys = ['matched', 'zero', 'half_matched', 'random']

    # build original weight matrix and convert to drive transition probability distribution

    ws = {}

    ws['matched'] = er_directed(N_NODES, P_CONNECT)
    p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(ws['matched'], G_W)

    # build the other weight matrices

    ws['zero'] = np.zeros((N_NODES, N_NODES), dtype=float)

    ws['half_matched'] = ws['matched'].copy()
    ws['half_matched'][np.random.rand(*ws['half_matched'].shape) < 0.5] = 0

    ws['random'] = er_directed(N_NODES, P_CONNECT)

    # make networks

    ntwks = {}

    for key, w in ws.items():
        ntwks[key] = SoftmaxWTAWithLingeringHyperexcitability(
            w, g_w=G_W, g_x=0, g_d=None, t_x=0)

    # perform a few checks

    assert np.sum(np.abs(ws['zero'])) == 0

    # create sample drive sequence

    drives = np.zeros((N_TIME_POINTS, N_NODES))

    drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
    drives[0, drive_first] = 1

    for ctr in range(N_TIME_POINTS - 1):
        drive_last = np.argmax(drives[ctr])
        drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

        drives[ctr + 1, drive_next] = 1

    drive_seq = np.argmax(drives, axis=1)
    drive_seqs = {
        seq_len: metrics.gather_sequences(drive_seq, seq_len)
        for seq_len in DECODING_SEQUENCE_LENGTHS
        }

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {
        key: {
            seq_len: []
            for seq_len in DECODING_SEQUENCE_LENGTHS
            }
        for key in keys
        }

    decoding_results_examples = {}

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for g_d in list(G_DS) + [G_D_EXAMPLE]:

        # set drive gain in all networks and run them

        for key, ntwk in ntwks.items():

            ntwk.g_d = g_d

            rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

            # calculate decoding accuracy for this network for all specified sequence lengths

            for seq_len in DECODING_SEQUENCE_LENGTHS:

                rs_seq_staggered = metrics.gather_sequences(rs_seq, seq_len)

                decoding_results = np.all(rs_seq_staggered == drive_seqs[seq_len], axis=1)

                decoding_accuracies[key][seq_len].append(np.mean(decoding_results))

            if g_d == G_D_EXAMPLE:

                decoding_results_examples[key] = (rs_seq == drive_seq)

    ## MAKE PLOTS FOR FIRST SIMULATION

    n_seq_lens = len(DECODING_SEQUENCE_LENGTHS)

    fig = plt.figure(figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    # the top row will be decoding accuracies for all sequence lengths

    axs = [
        fig.add_subplot(2 + len(MATCH_PROPORTION_SEQUENCE_LENGTHS), n_seq_lens, ctr + 1)
        for ctr in range(n_seq_lens)
    ]

    # the bottom row is example decoding accuracy time courses

    axs.append(fig.add_subplot(2 + len(MATCH_PROPORTION_SEQUENCE_LENGTHS), 1, 2))

    for ax, seq_len in zip(axs[:-1], DECODING_SEQUENCE_LENGTHS):

        for key, color in zip(keys, COLORS):
            ax.plot(G_DS, decoding_accuracies[key][seq_len][:-1], c=color, lw=2)

        ax.set_xlim(G_DS[0], G_DS[-1])
        ax.set_ylim(0, 1.1)

        ax.set_xlabel('g_d')

    axs[0].set_ylabel('decoding accuracy')

    for ax, seq_len in zip(axs[:-1], DECODING_SEQUENCE_LENGTHS):
        ax.set_title('Length {} sequences'.format(seq_len))

    axs[0].legend(keys, loc='best')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):
        decoding_results = decoding_results_examples[key]
        y_vals = 2 * ctr + decoding_results

        axs[-1].plot(y_vals, c=color, lw=2)
        axs[-1].axhline(2 * ctr, color='gray', lw=1, ls='--')

    axs[-1].set_xlim(0, N_TIME_POINTS_EXAMPLE)
    axs[-1].set_ylim(-1, 2 * len(keys) + 1)

    axs[-1].set_xlabel('time step')
    axs[-1].set_ylabel('correct decoding')

    axs[-1].set_title('example decoder time course')


    ## RUN SECOND SIMULATION -- VARIED MATCH PROPORTIONS

    keys = ['mixed_random', 'mixed_zero']

    # the following is indexed by [key][seq_len][trial][match_proportion_idx]

    decoding_accuracies = {
        key: {
            seq_len: [[] for _ in range(N_TRIALS)]
            for seq_len in DECODING_SEQUENCE_LENGTHS
            }
        for key in keys
        }

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for tr_ctr in range(N_TRIALS):

        # build original weight matrix and convert to drive transition probability distribution

        w_matched = er_directed(N_NODES, P_CONNECT)
        p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(w_matched, G_W)

        # build template random and zero matrices

        w_random = er_directed(N_NODES, P_CONNECT)
        w_zero = np.zeros((N_NODES, N_NODES), dtype=float)

        for mp_ctr, match_proportion in enumerate(MATCH_PROPORTIONS):

            ws = {}

            # make mixed weight matrices

            random_mask = np.random.rand(*w_matched.shape) < match_proportion
            zero_mask = np.random.rand(*w_matched.shape) < match_proportion

            ws['mixed_random'] = w_random.copy()
            ws['mixed_random'][random_mask] = w_matched[random_mask]

            ws['mixed_zero'] = w_zero.copy()
            ws['mixed_zero'][zero_mask] = w_matched[zero_mask]

            # make networks

            ntwks = {}

            for key, w in ws.items():
                ntwks[key] = SoftmaxWTAWithLingeringHyperexcitability(
                    w, g_w=G_W, g_x=0, g_d=G_D_EXAMPLE, t_x=0)

            # create random drive sequences

            drives = np.zeros((N_TIME_POINTS, N_NODES))

            drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
            drives[0, drive_first] = 1

            for ctr in range(N_TIME_POINTS - 1):
                drive_last = np.argmax(drives[ctr])
                drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

                drives[ctr + 1, drive_next] = 1

            drive_seq = np.argmax(drives, axis=1)
            drive_seqs = {
                seq_len: metrics.gather_sequences(drive_seq, seq_len)
                for seq_len in DECODING_SEQUENCE_LENGTHS
                }

            # run networks

            for key, ntwk in ntwks.items():

                rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

                # calculate decoding accuracy for this network for all specified sequence lengths

                for seq_len in DECODING_SEQUENCE_LENGTHS:

                    rs_seq_staggered = metrics.gather_sequences(rs_seq, seq_len)

                    decoding_results = np.all(rs_seq_staggered == drive_seqs[seq_len], axis=1)

                    decoding_accuracies[key][seq_len][tr_ctr].append(np.mean(decoding_results))

    ## MAKE PLOTS FOR SECOND SIMULATION

    n_seq_lens = len(MATCH_PROPORTION_SEQUENCE_LENGTHS)

    for ctr in range(n_seq_lens):

        axs.append(fig.add_subplot(2 + n_seq_lens, 1, 2 + ctr + 1))

    for ax, seq_len in zip(axs[-n_seq_lens:], MATCH_PROPORTION_SEQUENCE_LENGTHS):

        handles = []

        for ctr, (key, color) in enumerate(zip(keys, MATCH_PROPORTION_COLORS)):

            accs = np.array(decoding_accuracies[key][seq_len])

            accs_mean = accs.mean(axis=0)
            accs_sem = stats.sem(accs, axis=0)

            handles.append(ax.plot(MATCH_PROPORTIONS, accs_mean, c=color, lw=2, label=key, zorder=1)[0])

            ax.fill_between(
                MATCH_PROPORTIONS, accs_mean - accs_sem, accs_mean + accs_sem,
                color=color, alpha=0.1)

        ax.legend(handles=handles, loc='upper left')

        ax.set_xlim(0, 1)
        ax.set_ylim(-.1, 1.1)

        ax.set_xlabel('match proportion')
        ax.set_ylabel('decoding accuracy')

        ax.set_title('length {} sequences'.format(seq_len))

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig