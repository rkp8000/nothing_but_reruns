"""
Supplementary material.
"""
from __future__ import division, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import connectivity
import network
import plot
from shortcuts import reorder_idxs


def reverse_replay_lif_demo(
        SEED,
        SEQ_REVS, NODE_ORDER, NETWORK_SIZE_LIF, DT, OFFSET,
        W_PP, W_MP, W_PM, W_MM, W_PI, W_IP,
        TAU_P, TAU_M, TAU_I,
        V_REST_P, V_REST_M, V_REST_I,
        V_TH_P, V_TH_M, V_TH_I,
        V_RESET_P, V_RESET_M, V_RESET_I,
        RP_P, RP_M, RP_I,
        TAUS_SYN, V_REVS_SYN,
        SEQ_START, SEQ_DUR, SEQ_STAGGER, SEQ_FREQ, SEQ_AMP,
        REPLAY_START_FOR, REPLAY_START_REV, REPLAY_DUR, REPLAY_FREQ, REPLAY_AMP,
        RESET_START, RESET_DUR, RESET_AMP, RESET_FREQ,
        BKGD_GABAS):
    """
    Figure showing example of reverse replay in LIF network.
    """
    fig = plt.figure(figsize=(15, 17), tight_layout=True)
    gs = gridspec.GridSpec(7, 4)
    np.random.seed(SEED)

    # build weight matrices
    w_base, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE_LIF)
    # reorder nodes for nicer graphical presentation
    nodes, idxs = reorder_idxs(nodes, NODE_ORDER)
    w_base = w_base[idxs, :][:, idxs]

    n = len(nodes)

    w_pp = W_PP * w_base
    w_pm = W_PM * np.eye(n)
    w_mm = W_MM * np.eye(n)
    w_mp = W_MP * np.eye(n)
    w_pi = W_PI * np.ones((n,))
    w_ip = W_IP * np.ones((n,))

    w_ampa = np.zeros((2*n+1, 2*n+1))
    w_nmda = np.zeros((2*n+1, 2*n+1))
    w_gaba = np.zeros((2*n+1, 2*n+1))

    w_ampa[n:2*n, :n] = w_mp
    w_ampa[-1, :n] = w_ip

    w_nmda[:n, :n] = w_pp
    w_nmda[:n, n:2*n] = w_pm
    w_nmda[n:2*n, n:2*n] = w_mm

    w_gaba[:n, -1] = w_pi

    ws = {'ampa': w_ampa, 'nmda': w_nmda, 'gaba': w_gaba}

    # build network
    cc = np.concatenate
    taus_m = cc([TAU_P * np.ones((n,)), TAU_M * np.ones((n,)), [TAU_I]])
    v_rests = cc([V_REST_P * np.ones((n,)), V_REST_M * np.ones((n,)), [V_REST_I]])
    v_ths = cc([V_TH_P * np.ones((n,)), V_TH_M * np.ones((n,)), [V_TH_I]])
    v_resets = cc([V_RESET_P * np.ones((n,)), V_RESET_M * np.ones((n,)), [V_RESET_I]])
    refrac_pers = cc([RP_P * np.ones((n,)), RP_M * np.ones((n,)), [RP_I]])

    ntwk = network.LIFExponentialSynapsesModel(
        taus_m=taus_m, v_rests=v_rests, v_ths=v_ths, v_resets=v_resets,
        refrac_pers=refrac_pers, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN, ws=ws)

    # build stimulus
    dt = DT
    dur = 2*OFFSET
    n_steps = int(dur/dt)

    drives = {syn: np.zeros((n_steps, 2*n+1)) for syn in ws.keys()}

    seq_start = int(SEQ_START/dt)
    seq_dur = int(SEQ_DUR/dt)
    seq_stagger = int(SEQ_STAGGER/dt)

    replay_starts = [int(REPLAY_START_FOR/dt), int(REPLAY_START_REV/dt)]
    replay_dur = int(REPLAY_DUR/dt)

    offsets = [0, int(OFFSET/dt)]
    seq_orders = [[nodes.index(node) for node in seq] for seq in SEQ_REVS]

    for offset, seq_order in zip(offsets, seq_orders):
        # excitatory drives

        ## initial sequence
        for ctr, node_idx in enumerate(seq_order):
            start = offset + seq_start + seq_stagger*ctr
            end = start + seq_dur
            inputs = SEQ_AMP * \
                (np.random.rand(end-start) < (SEQ_FREQ*dt)).astype(float)

            drives['ampa'][start:end, node_idx] = inputs

        ## replay triggers
        triggers = [seq_order[0], seq_order[-1]]
        for replay_start, trigger in zip(replay_starts, triggers):
            start = offset + replay_start
            end = start + replay_dur
            inputs = REPLAY_AMP * \
                (np.random.rand(end-start) < (REPLAY_FREQ*dt)).astype(float)

            drives['ampa'][start:end, trigger] = inputs

        # inhibitory drives
        ## reset
        start = offset + int(RESET_START/dt)
        end = start + int(RESET_DUR/dt)
        inputs = RESET_AMP * \
            (np.random.rand(end-start, n) < (RESET_FREQ*dt)).astype(float)

        drives['gaba'][start:end, n:2*n] = inputs

        ## background gaba
        for bkgd_gaba in BKGD_GABAS:
            start = offset + int(bkgd_gaba['start']/dt)
            end = offset + int(bkgd_gaba['end']/dt)

            amp = bkgd_gaba['amp']
            freq = bkgd_gaba['freq']
            inputs = amp * (np.random.rand(end-start, n) < (freq*dt)).astype(float)

            drives['gaba'][start:end, :n] = inputs

    # set initial network conditions
    initial_conditions = {
        'voltages': v_rests,
        'conductances': {syn: np.zeros((2*n+1,)) for syn in ws.keys()},
        'refrac_ctrs': np.zeros((2*n+1,)),
    }

    # run network
    measurements = ntwk.run(
        initial_conditions=initial_conditions, drives=drives, dt=dt,
        record=('voltages', 'spikes'))

    # make plots
    axs = [fig.add_subplot(gs[ctr:ctr+2, :]) for ctr in [0, 2]]
    axs.append(fig.add_subplot(gs[4:, :]))
    marker_size = 15

    handles = []

    # plot ampa drives to primary neurons
    drive_times, drive_idxs = drives['ampa'][:, :n].nonzero()
    drive_times = dt * drive_times.astype(float)

    handles.append(axs[0].scatter(
        drive_times, drive_idxs + 2*n,
        marker='|', s=marker_size, lw=1.5, c='b', label='AMPA'))

    # plot gaba drives to memory neurons
    drive_times, drive_idxs = drives['gaba'][:, n:2*n].nonzero()
    drive_times = dt * drive_times.astype(float)

    handles.append(axs[0].scatter(
        drive_times, drive_idxs + 4, marker='|', s=marker_size, c='r', label='NMDA'))

    # plot primary spikes
    spike_times, spike_idxs = measurements['spikes'][:, :n].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(
        spike_times, spike_idxs + 2*n, marker='|', lw=1.3, s=marker_size, c='k')

    # plot memory spikes
    spike_times, spike_idxs = measurements['spikes'][:, n:2*n].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(spike_times, spike_idxs + 4, marker='|', s=marker_size, c='g')

    # plot inhibitory spikes
    spike_times, spike_idxs = measurements['spikes'][:, [2*n]].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(spike_times, spike_idxs, marker='|', s=marker_size, c='r')

    axs[0].set_xlim(0, dt*n_steps)
    axs[1].set_xlim(0, dt*n_steps)

    for ax in axs:
        ax.set_ylabel('neuron')

    axs[0].set_title('stimulus')
    axs[1].set_title('spikes')
    axs[1].set_xlabel('time (s)')

    # DEBUGGING: plot voltage of first three neurons in first sequence
    ts = np.arange(len(measurements['voltages'])) * dt
    y_ticks = []
    y_tick_labels = []

    for ctr, node in enumerate(SEQ_REVS[0][:6] + [(-2, 0)]):
        offset = -.05 * ctr  # 50 mV spacing on plot between traces
        axs[2].axhline(offset + V_REVS_SYN['gaba'], color='gray', ls='--')
        axs[2].axhline(offset + V_TH_P, color='gray', ls='--')
        axs[2].axhline(offset + V_RESET_P, color='r', ls='--')

        vs = measurements['voltages'][:, nodes.index(node)]
        spike_times = dt*measurements['spikes'][:, nodes.index(node)].nonzero()[0]
        axs[2].plot(ts, offset + vs, color='k')

        axs[2].scatter(
            spike_times, (offset+V_TH_P) * np.ones(spike_times.shape),
            c='k', s=50, lw=0, zorder=100)

        y_ticks.append(offset + .5 * (V_RESET_P + V_TH_P))
        y_tick_labels.append('node {}'.format(node))

    axs[2].set_xlim(1.5, 2.5)
    axs[2].set_yticks(y_ticks)
    axs[2].set_yticklabels(y_tick_labels)

    axs[2].set_xlabel('time (s)')

    for ax in axs: plot.set_fontsize(ax, 16)

    return fig
