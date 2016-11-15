"""
Figures for Cosyne 2017 abstract.
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

from connectivity import hexagonal_lattice
from network import BasicWithAthAndTwoLevelStdp
from network import LIFExponentialSynapsesModel
from plot import get_n_colors, set_fontsize


def toy_network_simplified(TH, G_W, G_X, T_X, RP):
    """
    Demonstrate replay in a toy network.
    """

    # build weight matrix
    w = G_W * np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ], dtype=float)

    ntwk = BasicWithAthAndTwoLevelStdp(TH, w, G_X, T_X, RP, stdp_params=None)

    # make stimulus
    drives = np.zeros((4 * T_X, 7), dtype=float)

    # first sequence encoding
    drives[2, 0] = 1
    drives[3, 1] = 1
    drives[4, 2] = 1
    drives[5, 3] = 1
    drives[6, 4] = 1

    # first sequence replay
    drives[8, 0] = 1

    # second sequence encoding
    drives[2 * T_X + 2, 0] = 1
    drives[2 * T_X + 3, 1] = 1
    drives[2 * T_X + 4, 2] = 1
    drives[2 * T_X + 5, 5] = 1
    drives[2 * T_X + 6, 6] = 1

    # second sequence replay
    drives[2 * T_X + 8, 0] = 1

    # run network
    r_0 = np.zeros((7,))
    xc_0 = np.zeros((7,))
    rs, _ = ntwk.run(r_0, xc_0, 5*drives)

    # set up figures
    fig, axs = plt.subplots(2, 1, figsize=(3, 4), tight_layout=True, sharex=True, sharey=True)

    # plot drives
    drive_times, drive_idxs = drives.nonzero()
    axs[0].scatter(drive_times, drive_idxs, c='b', lw=0)
    axs[0].set_title('stimulus')

    # plot spikes
    spike_times, spike_idxs = rs.nonzero()
    axs[1].scatter(spike_times, spike_idxs, c='k', lw=0)

    axs[1].set_xlabel('time step')
    axs[1].set_title('spikes')

    for ax in axs:

        ax.set_ylabel('unit')
        ax.set_yticks([0, 2, 4, 6])
        ax.set_yticklabels([1, 3, 5, 7])
        set_fontsize(ax, 12)

    return fig


def toy_network_lif(
        SEED, DT, OFFSET,
        W_PP, W_MP, W_PM, W_MM, W_PI, W_IP,
        TAU_P, TAU_M, TAU_I,
        V_REST_P, V_REST_M, V_REST_I,
        V_TH_P, V_TH_M, V_TH_I,
        V_RESET_P, V_RESET_M, V_RESET_I,
        RP_P, RP_M, RP_I,
        TAUS_SYN, V_REVS_SYN,
        SEQ_START, SEQ_DUR, SEQ_STAGGER, SEQ_FREQ, SEQ_AMP,
        REPLAY_START, REPLAY_DUR, REPLAY_FREQ, REPLAY_AMP,
        RESET_START, RESET_DUR, RESET_AMP, RESET_FREQ,
        BKGD_GABA_AMP, BKGD_GABA_FREQ,
        BKGD_GABA_AMP_MEM, BKGD_GABA_FREQ_MEM):
    """
    Demonstrate replay in a toy network.
    """

    np.random.seed(SEED)

    # build weight matrices
    w_base = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ], dtype=float)

    w_pp = W_PP * w_base
    w_pm = W_PM * np.eye(7)
    w_mm = W_MM * np.eye(7)
    w_mp = W_MP * np.eye(7)
    w_pi = W_PI * np.ones((7,))
    w_ip = W_IP * np.ones((7,))

    w_ampa = np.zeros((15, 15))
    w_nmda = np.zeros((15, 15))
    w_gaba = np.zeros((15, 15))

    w_ampa[7:14, :7] = w_mp
    w_ampa[-1, :7] = w_ip

    w_nmda[:7, :7] = w_pp
    w_nmda[:7, 7:14] = w_pm
    w_nmda[7:14, 7:14] = w_mm

    w_gaba[:7, -1] = w_pi

    ws = {'ampa': w_ampa, 'nmda': w_nmda, 'gaba': w_gaba}

    # build network
    cc = np.concatenate
    taus_m = cc([TAU_P * np.ones((7,)), TAU_M * np.ones((7,)), [TAU_I]])
    v_rests = cc([V_REST_P * np.ones((7,)), V_REST_M * np.ones((7,)), [V_REST_I]])
    v_ths = cc([V_TH_P * np.ones((7,)), V_TH_M * np.ones((7,)), [V_TH_I]])
    v_resets = cc([V_RESET_P * np.ones((7,)), V_RESET_M * np.ones((7,)), [V_RESET_I]])
    refrac_pers = cc([RP_P * np.ones((7,)), RP_M * np.ones((7,)), [RP_I]])

    ntwk = LIFExponentialSynapsesModel(
        taus_m=taus_m, v_rests=v_rests, v_ths=v_ths, v_resets=v_resets, refrac_pers=refrac_pers,
        taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN, ws=ws)

    # build stimulus
    dt = DT
    dur = 2*OFFSET
    n_steps = int(dur/dt)

    drives = {syn: np.zeros((n_steps, 15)) for syn in ws.keys()}

    seq_start = int(SEQ_START/dt)
    seq_dur = int(SEQ_DUR/dt)
    seq_stagger = int(SEQ_STAGGER/dt)

    replay_start = int(REPLAY_START/dt)
    replay_dur = int(REPLAY_DUR/dt)

    seq_order_0 = (0, 1, 2, 3, 4)
    seq_order_1 = (0, 1, 2, 5, 6)

    for offset, seq_order in zip((0, int(OFFSET/dt)), (seq_order_0, seq_order_1)):
        # excitatory drives
        for ctr, node in enumerate(seq_order):

            start = offset + seq_start + seq_stagger*ctr
            end = start + seq_dur
            inputs = SEQ_AMP * (np.random.rand(end-start) < (SEQ_FREQ*dt)).astype(float)

            drives['ampa'][start:end, node] = inputs

        start = offset + replay_start
        end = start + replay_dur
        inputs = REPLAY_AMP * (np.random.rand(end-start) < (REPLAY_FREQ*dt)).astype(float)

        drives['ampa'][start:end, seq_order[0]] = inputs

        # inhibitory drives

        ## reset
        start = offset + int(RESET_START/dt)
        end = start + int(RESET_DUR/dt)
        inputs = RESET_AMP * (np.random.rand(end-start, 7) < (RESET_FREQ*dt)).astype(float)

        drives['gaba'][start:end, 7:14] = inputs

        ## background
        start = offset
        end = offset + seq_start + seq_stagger * (len(seq_order) - 1) + seq_dur + int(0.2/dt)
        inputs = BKGD_GABA_AMP * (np.random.rand(end-start, 7) < (BKGD_GABA_FREQ*dt)).astype(float)

        drives['gaba'][start:end, :7] = inputs

    # inhibitory background to memory units
    drives['gaba'][:, 7:14] += \
        BKGD_GABA_AMP_MEM * (np.random.rand(len(drives['gaba']), 7) < (BKGD_GABA_FREQ_MEM*dt)).astype(float)

    # set initial network conditions
    initial_conditions = {
        'voltages': v_rests,
        'conductances': {syn: np.zeros((15,)) for syn in ws.keys()},
        'refrac_ctrs': np.zeros((15,)),
    }

    # run network
    measurements = ntwk.run(
        initial_conditions=initial_conditions, drives=drives, dt=dt, record=('voltages', 'spikes'))

    # make plots
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True, sharey=True, tight_layout=True)
    marker_size = 15

    handles = []

    # plot ampa drives to primary neurons
    drive_times, drive_idxs = drives['ampa'][:, :7].nonzero()
    drive_times = dt * drive_times.astype(float)

    handles.append(axs[0].scatter(
        drive_times, drive_idxs + 14, marker='|', s=marker_size, lw=1.5, c='b', label='AMPA'))

    # plot gaba drives to memory neurons
    drive_times, drive_idxs = drives['gaba'][:, 7:14].nonzero()
    drive_times = dt * drive_times.astype(float)

    handles.append(axs[0].scatter(
        drive_times, drive_idxs + 4, marker='|', s=marker_size, c='r', label='NMDA'))

    # plot primary spikes
    spike_times, spike_idxs = measurements['spikes'][:, :7].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(spike_times, spike_idxs + 14, marker='|', lw=1.3, s=marker_size, c='k')

    # plot memory spikes
    spike_times, spike_idxs = measurements['spikes'][:, 7:14].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(spike_times, spike_idxs + 4, marker='|', s=marker_size, c='g')

    # plot inhibitory spikes
    spike_times, spike_idxs = measurements['spikes'][:, [14]].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(spike_times, spike_idxs, marker='|', s=marker_size, c='r')

    for ax in axs:
        ax.set_xlim(0, 8)
        ax.set_ylim(-1, 21)
        ax.set_yticks([0, 4, 7, 10, 14, 17, 20])
        ax.set_yticklabels(['I1', 'M1', 'M4', 'M7', 'P1', 'P4', 'P7'])
        ax.set_ylabel('neuron')

    axs[0].set_title('stimulus')
    axs[1].set_title('spikes')
    axs[1].set_xlabel('time (s)')

    for ax in axs: set_fontsize(ax, 12)

    return fig


def _get_stationary_distribution(trs):
    """
    Return stationary distribution given a transition matrix.
    :param trs:
    :return:
    """

    evs, evecs = np.linalg.eig(trs)

    idx = np.argmin(np.abs(np.abs(evs) - 1))
    if evs[idx] < 1: evecs *= -1

    p_0 = np.real(evecs[:, idx])
    p_0 /= p_0.sum()

    return p_0


def _sample_markov_chain(p_0, trs, l):
    """
    Sample a sequence from a Markov chain.
    :param p_0:
    :param trs:
    :param l:
    :return:
    """
    nodes = np.arange(len(p_0))
    seq = [np.random.choice(nodes, p=p_0)]

    for _ in range(l-1):
        seq.append(np.random.choice(nodes, p=trs[:, seq[-1]]))

    return seq


def replay_statistics(
        SEED,
        LS, N, N_TRIALS, N_STIM_SEQS, Q_REPLAY_PROB,
        G_W, G_X, RP, TH,
        QS, QS_MEMORY_CONTENT, NS_MEMORY_CONTENT):
    """
    Show replay statistics:
        Relative capacity vs. density for ER networks.
        Optimal density vs. L.
        Replay probability vs. percent stimulus-matched transitions.
        Guaranteed replay probability and memory content vs. sparsity.
    """

    np.random.seed(SEED)

    fig = plt.figure(figsize=(10, 4), tight_layout=True)
    axs = [fig.add_subplot(1, 3, 1)]
    axs.append(fig.add_subplot(1, 3, 2, sharey=axs[0]))
    axs.append(fig.add_subplot(1, 3, 3))

    colors = get_n_colors(len(LS) + 1, 'hsv')[:-1]

    # plot replay probability vs. stimulus-matched connectivity percentage
    match_percentages = np.linspace(0, 1, 11, endpoint=True)
    handles = []
    for l, color in zip(LS, colors):

        candidate_file_name = 'replay_probability_L_{}.npy'.format(l)
        if os.path.isfile(candidate_file_name):
            replay_probs = np.load(candidate_file_name)
        else:

            replay_probs = np.nan * np.zeros((N_TRIALS, len(match_percentages)))

            for trial_ctr in range(N_TRIALS):

                # generate random stimulus transition matrix
                while True:
                    trs = (np.random.rand(N, N) < Q_REPLAY_PROB).astype(float)
                    np.fill_diagonal(trs, 0)
                    if np.all(trs.sum(axis=0) > 0): break

                w_stim = trs.copy()

                # normalize all columns to 1 to make it probabilistic
                for col_ctr in range(N): trs[:, col_ctr] /= trs[:, col_ctr].sum()
                p_0 = _get_stationary_distribution(trs)

                # loop over match percentages
                w_rand = (np.random.rand(N, N) < Q_REPLAY_PROB).astype(float)

                for mp_ctr, mp in enumerate(match_percentages):

                    w = w_rand.copy()
                    mask = np.random.rand(*w.shape) < mp
                    w[mask] = w_stim[mask]
                    w *= G_W

                    # make network
                    ntwk = BasicWithAthAndTwoLevelStdp(
                        th=TH, w=w, g_x=G_X, t_x=2 * l, rp=RP, stdp_params=None
                    )

                    correct_ctr = 0

                    for _ in range(N_STIM_SEQS):

                        drives = np.zeros((2 * l + 2, N))
                        seq = _sample_markov_chain(p_0, trs, l)
                        for ctr, node in enumerate(seq):

                            drives[ctr + 1, node] = 1

                        drives[l + 2, seq[0]] = 1

                        r_0 = np.zeros((N,))
                        xc_0 = np.zeros((N,))

                        rs, _ = ntwk.run(r_0, xc_0, 5*drives)

                        if np.all(rs[l+2:2*l+2, :] == drives[1:l+1, :]): correct_ctr += 1

                    replay_probs[trial_ctr, mp_ctr] = correct_ctr / N_STIM_SEQS

            np.save(candidate_file_name, replay_probs)

        mean = np.mean(replay_probs, axis=0)
        sem = stats.sem(replay_probs, axis=0)
        handles.append(axs[0].plot(
            match_percentages, mean, color=color, lw=2, label='L = {}'.format(l), zorder=1)[0])
        axs[0].fill_between(
            match_percentages, mean-sem, mean+sem, color=color, zorder=0, alpha=0.2)

        axs[0].set_xticks([0, .2, .4, .6, .8, 1])
        axs[0].set_xticklabels(['0', '.2', '.4', '.6', '.8', '1'])

        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1.1)
        axs[0].set_xlabel('match proportion')
        axs[0].set_ylabel('p(correct replay)')
        axs[0].legend(handles=handles, loc='best')

    # plot replay probability and memory content
    ls_qs = ['-', '--', '-.']

    for l, color in zip(LS, colors):

        guaranteed_replay_probability = \
            (1 - QS) ** ((l - 1) * (l - 2))

        axs[1].semilogx(QS, guaranteed_replay_probability, color=color, lw=2)

        for q, ls in zip(QS_MEMORY_CONTENT, ls_qs):

            memory_content = []
            for n in NS_MEMORY_CONTENT:

                memory_content.append(np.prod(n - np.arange(1, l)) * (q ** (l-1)))

            memory_content = np.array(memory_content)
            memory_content[memory_content < 1] = np.nan

            axs[2].loglog(NS_MEMORY_CONTENT, memory_content, color=color, ls=ls, lw=2)

    axs[1].set_xlabel('density')

    axs[2].set_xlabel('N')
    axs[2].set_ylabel('memory content')

    axs[2].set_yticks(axs[2].get_yticks()[::4])

    for ax in axs: set_fontsize(ax, 14)

    return fig


def reverse_replay_simplified(TH, G_W, G_X, T_X, RP):
    """
    Show reverse replay in simplified.
    """

    # build weight matrix and network
    w, nodes = hexagonal_lattice(4)
    w *= G_W

    n_nodes = len(nodes)

    ntwk = BasicWithAthAndTwoLevelStdp(TH, w, G_X, T_X, RP, stdp_params=None)

    # build stimulus
    drives = np.zeros((6 * T_X, n_nodes))

    # first sequence
    drives[2, nodes.index((0, 3))] = 1
    drives[3, nodes.index((0, 5))] = 1
    drives[4, nodes.index((0, 7))] = 1
    drives[5, nodes.index((0, 9))] = 1
    drives[6, nodes.index((1, 10))] = 1
    drives[7, nodes.index((2, 11))] = 1
    drives[8, nodes.index((3, 12))] = 1

    # first forward replay
    drives[12, nodes.index((0, 3))] = 1

    # first reverse replay
    drives[22, nodes.index((3, 12))] = 1

    # second sequence
    drives[3 * T_X + 2, nodes.index((0, 3))] = 1
    drives[3 * T_X + 3, nodes.index((1, 4))] = 1
    drives[3 * T_X + 4, nodes.index((2, 5))] = 1
    drives[3 * T_X + 5, nodes.index((3, 6))] = 1
    drives[3 * T_X + 6, nodes.index((3, 8))] = 1
    drives[3 * T_X + 7, nodes.index((3, 10))] = 1
    drives[3 * T_X + 8, nodes.index((3, 12))] = 1

    # second sequence replay
    drives[3 * T_X + 12, nodes.index((0, 3))] = 1

    # second reverse replay
    drives[3 * T_X + 22, nodes.index((3, 12))] = 1

    # run network
    r_0 = np.zeros((n_nodes,))
    xc_0 = np.zeros((n_nodes,))
    rs, _ = ntwk.run(r_0, xc_0, 5*drives)

    fig, axs = plt.subplots(2, 1, figsize=(4.5, 4), tight_layout=True, sharex=True, sharey=True)

    # plot drives
    marker_size = 12
    drive_times, drive_idxs = drives.nonzero()
    axs[0].scatter(drive_times, drive_idxs, s=marker_size, c='b', lw=0)
    axs[0].set_title('stimulus')

    # plot spikes
    spike_times, spike_idxs = rs.nonzero()
    axs[1].scatter(spike_times, spike_idxs, s=marker_size, c='k', lw=0)

    axs[1].set_xlabel('time step')
    axs[1].set_title('spikes')

    ticks = np.arange(0, n_nodes, n_nodes // 5)

    for ax in axs:
        ax.set_xlim(-1, 85)
        ax.set_ylim(-3, 25)
        ax.set_ylabel('unit')
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks + 1)
        set_fontsize(ax, 14)

    axs[1].set_ylim(-3, 25)

    return fig


def reverse_replay_lif(
        SEED,
        TAU_P, TAU_M, TAU_I,
        V_REST_P, V_REST_M, V_REST_I,
        V_RESET_P, V_RESET_M, V_RESET_I,
        V_TH_P, V_TH_M, V_TH_I,
        TAUS_SYN, V_REVS_SYN,
        RP_P, RP_M, RP_I):
    """
    Show reverse replay in simplified and LIF network.
    """

    pass
