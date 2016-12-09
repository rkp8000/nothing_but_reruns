from __future__ import division, print_function
from itertools import product as cproduct
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np
from scipy import stats

import connectivity
from db import _models
import db
import network
import plot
from shortcuts import make_drive_seq, zip_cproduct, reorder_idxs
from shortcuts import get_stationary_distribution, sample_markov_chain


def replay_demo_simplified_and_lif(
        # simplified network
        V_TH, G_W, G_X, T_X, RP,
        # lif network
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
    Demonstrate activation-triggered-hyperexcitability-mediated sequence
    replay in simplified and LIF network.
    """
    fig = plt.figure(figsize=(15, 15), tight_layout=True)
    gs = gridspec.GridSpec(6, 3)

    # simplified model
    fig.add_subplot(gs[0, :])  # diagrams will go here
    axs = [fig.add_subplot(gs[ctr, :]) for ctr in range(1, 3)]

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

    ntwk = network.BasicWithAthAndTwoLevelStdp(V_TH, w, G_X, T_X, RP, stdp_params=None)

    # make stimulus
    drives = np.zeros((5 * T_X, 7), dtype=float)

    # first sequence encoding
    drives[1, 0] = 1
    drives[2, 1] = 1
    drives[3, 2] = 1
    drives[4, 3] = 1
    drives[5, 4] = 1

    # first sequence replay
    drives[T_X, 0] = 1

    # second sequence encoding
    drives[3 * T_X + 1, 0] = 1
    drives[3 * T_X + 2, 1] = 1
    drives[3 * T_X + 3, 2] = 1
    drives[3 * T_X + 4, 5] = 1
    drives[3 * T_X + 5, 6] = 1

    # second sequence replay
    drives[4 * T_X, 0] = 1

    # run network
    r_0 = np.zeros((7,))
    xc_0 = np.zeros((7,))
    rs, _ = ntwk.run(r_0, xc_0, 5*drives)

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
        ax.set_xlim(0, len(drives))

    for ax in axs:
        ax.set_ylabel('unit')
        ax.set_yticks([0, 2, 4, 6])
        ax.set_yticklabels([1, 3, 5, 7])
        plot.set_fontsize(ax, 16)

    # lif implementation
    axs = [fig.add_subplot(gs[ctr, :]) for ctr in range(4, 6)] + \
        [fig.add_subplot(gs[3, 2])]

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

    ntwk = network.LIFExponentialSynapsesModel(
        taus_m=taus_m, v_rests=v_rests, v_ths=v_ths, v_resets=v_resets,
        refrac_pers=refrac_pers, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN, ws=ws)

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
            inputs = SEQ_AMP * \
                (np.random.rand(end-start) < (SEQ_FREQ*dt)).astype(float)

            drives['ampa'][start:end, node] = inputs

        start = offset + replay_start
        end = start + replay_dur
        inputs = REPLAY_AMP * \
            (np.random.rand(end-start) < (REPLAY_FREQ*dt)).astype(float)

        drives['ampa'][start:end, seq_order[0]] = inputs

        # inhibitory drives
        ## reset
        start = offset + int(RESET_START/dt)
        end = start + int(RESET_DUR/dt)
        inputs = RESET_AMP * \
            (np.random.rand(end-start, 7) < (RESET_FREQ*dt)).astype(float)

        drives['gaba'][start:end, 7:14] = inputs

        ## background
        start = offset
        end = offset + seq_start + seq_stagger * (len(seq_order) - 1) + \
              seq_dur + int(0.2/dt)
        inputs = BKGD_GABA_AMP * \
            (np.random.rand(end-start, 7) < (BKGD_GABA_FREQ*dt)).astype(float)

        drives['gaba'][start:end, :7] = inputs

    # inhibitory background to memory units
    drives['gaba'][:, 7:14] += BKGD_GABA_AMP_MEM * \
        (np.random.rand(len(drives['gaba']), 7) < (BKGD_GABA_FREQ_MEM*dt)).\
        astype(float)

    # set initial network conditions
    initial_conditions = {
        'voltages': v_rests,
        'conductances': {syn: np.zeros((15,)) for syn in ws.keys()},
        'refrac_ctrs': np.zeros((15,)),
    }

    # run network
    measurements = ntwk.run(
        initial_conditions=initial_conditions, drives=drives, dt=dt,
        record=('voltages', 'spikes'))

    # make plots
    marker_size = 15

    handles = []

    # plot ampa drives to primary neurons
    drive_times, drive_idxs = drives['ampa'][:, :7].nonzero()
    drive_times = dt * drive_times.astype(float)

    handles.append(axs[0].scatter(
        drive_times, drive_idxs + 14,
        marker='|', s=marker_size, lw=1.5, c='b', label='AMPA'))

    # plot gaba drives to memory neurons
    drive_times, drive_idxs = drives['gaba'][:, 7:14].nonzero()
    drive_times = dt * drive_times.astype(float)

    handles.append(axs[0].scatter(
        drive_times, drive_idxs + 4, marker='|', s=marker_size, c='r', label='NMDA'))

    # plot primary spikes
    spike_times, spike_idxs = measurements['spikes'][:, :7].nonzero()
    spike_times = dt * spike_times.astype(float)

    axs[1].scatter(
        spike_times, spike_idxs + 14, marker='|', lw=1.3, s=marker_size, c='k')

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

    for ax in axs: plot.set_fontsize(ax, 16)

    return fig


def record_connectivity_analysis(
        SEED, GROUP, LOG_FILE,
        V_TH, G_W, G_X, RP,
        N, LS, QS, MATCH_PERCENTS, N_TRIALS, N_STIM_SEQS):
    """
    Analyze the dependence of replay probability on the percent match between
    the stimulus transition matrix and network connectivity.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    db.prepare_logging(LOG_FILE)
    np.random.seed(SEED)

    for l, q in cproduct(LS, QS):

        logging.info('Running sim. for L = {}, Q = {}'.format(l, q))
        replay_probs = np.nan * np.zeros((N_TRIALS, len(MATCH_PERCENTS)))

        for trial_ctr in range(N_TRIALS):

            logging.info('Trial {} started.'.format(trial_ctr + 1))

            # generate random stimulus transition matrix
            while True:
                trs = (np.random.rand(N, N) < q).astype(float)
                np.fill_diagonal(trs, 0)
                if np.all(trs.sum(axis=0) > 0): break

            w_stim = trs.copy()

            # normalize all columns to 1 to make it probabilistic
            for col_ctr in range(N):
                trs[:, col_ctr] /= trs[:, col_ctr].sum()

            p_0 = get_stationary_distribution(trs)

            # loop over match percentages
            w_rand = (np.random.rand(N, N) < q).astype(float)

            for mp_ctr, mp in enumerate(MATCH_PERCENTS):

                w = w_rand.copy()
                mask = np.random.rand(*w.shape) < mp
                w[mask] = w_stim[mask]
                w *= G_W

                # make network
                ntwk = network.BasicWithAthAndTwoLevelStdp(
                    th=V_TH, w=w, g_x=G_X, t_x=2 * l, rp=RP, stdp_params=None)

                correct_ctr = 0

                for _ in range(N_STIM_SEQS):

                    drives = np.zeros((2 * l + 2, N))
                    seq = sample_markov_chain(p_0, trs, l)
                    for ctr, node in enumerate(seq):

                        drives[ctr + 1, node] = 1

                    drives[l + 2, seq[0]] = 1

                    r_0 = np.zeros((N,))
                    xc_0 = np.zeros((N,))

                    rs, _ = ntwk.run(r_0, xc_0, 5*drives)

                    if np.all(rs[l+2:2*l+2, :] == drives[1:l+1, :]): correct_ctr += 1

                replay_probs[trial_ctr, mp_ctr] = correct_ctr / N_STIM_SEQS

        car = _models.ConnectivityAnalysisResult(
            group=GROUP,
            n=N, l=l, q=q,
            match_percents=MATCH_PERCENTS,
            n_trials=N_TRIALS, n_stim_seqs=N_STIM_SEQS,
            v_th=V_TH, g_w=G_W, g_x=G_X, rp=RP,
            replay_probs=replay_probs.tolist())

        session.add(car)
        session.commit()
    session.close()


def capacity_and_connectivity_analysis(
        GROUP, QS_MATCH_ANALYSIS,
        LS, QS_DENSITY, QS_MEMORY_CONTENT, NS_MEMORY_CONTENT):
    """
    Show replay statistics:
        Relative capacity vs. density for ER networks.
        Optimal density vs. L.
        Replay probability vs. percent stimulus-matched transitions.
        Guaranteed replay probability and memory content vs. sparsity.
    """
    session = db.connect_and_make_session('nothing_but_reruns')

    fig = plt.figure(figsize=(10, 4), tight_layout=True)
    axs = [fig.add_subplot(1, 3, 1)]
    axs.append(fig.add_subplot(1, 3, 2, sharey=axs[0]))
    axs.append(fig.add_subplot(1, 3, 3))

    styles = ['-', '--', '-.']

    # plot replay probability vs. stimulus-matched connectivity percentage
    handles = []
    for q, style in zip(QS_MATCH_ANALYSIS, styles):

        cars = session.query(_models.ConnectivityAnalysisResult).filter(
            _models.ConnectivityAnalysisResult.group == GROUP,
            _models.ConnectivityAnalysisResult.q.between(.99 * q, 1.01*q))

        ls = sorted(np.unique([car.l for car in cars.all()]))
        colors = get_n_colors(len(ls) + 1, 'hsv')[:-1]

        for l, color in zip(ls, colors):

            car = cars.filter(_models.ConnectivityAnalysisResult.l == l).first()
            replay_probs = np.array(car.replay_probs)

            mean = np.mean(replay_probs, axis=0)
            sem = stats.sem(replay_probs, axis=0)
            handles.append(axs[0].plot(
                match_percentages, mean, color=color, lw=2, ls=style,
                label='L = {}'.format(l), zorder=1)[0])
            axs[0].fill_between(
                match_percentages, mean-sem, mean+sem,
                color=color, zorder=0, alpha=0.2)

            axs[0].set_xticks([0, .2, .4, .6, .8, 1])
            axs[0].set_xticklabels(['0', '.2', '.4', '.6', '.8', '1'])

            axs[0].set_xlim(0, 1)
            axs[0].set_ylim(0, 1.1)
            axs[0].set_xlabel('match proportion')
            axs[0].set_ylabel('p(correct replay)')
            axs[0].legend(handles=handles, loc='best')

    # plot replay probability and memory content
    colors = get_n_colors(len(ls) + 1, 'hsv')[:-1]
    for l, color in zip(LS, colors):

        guaranteed_replay_probability = \
            (1 - QS) ** ((l - 1) * (l - 2))

        axs[1].semilogx(QS_DENSITY, guaranteed_replay_probability, color=color, lw=2)

        for q, ls in zip(QS_MEMORY_CONTENT, styles):

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


def multiple_and_reverse_replay(
        # simplified model parameters
        SEED_SIMPLIFIED, NETWORK_SIZE, V_TH, G_W, G_X, T_X, RP,
        # multiple replay simplified demo
        SEQ_REF, SEQ_NONOVERLAP, SEQ_OVERLAP,
        TRIGGER_START, TRIGGER_END,
        TRIGGER_INTERVAL,
        NODE_ORDER_MULTIPLE,
        # reverse replay simplified demo
        SEQ_REVS, NODE_ORDER_REVERSE,
        # lif network
        SEED_LIF, NETWORK_SIZE_LIF, DT, OFFSET,
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
    Show figures displaying replay of multiple sequences, priming, and
    reverse replay.
    """
    fig = plt.figure(figsize=(15, 27), tight_layout=True)
    gs = gridspec.GridSpec(11, 4)
    np.random.seed(SEED_SIMPLIFIED)

    # build simplified network
    w, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE)
    nodes, idxs = reorder_idxs(nodes, NODE_ORDER_MULTIPLE)
    w = w[idxs, :][:, idxs]
    ntwk = network.LocalWtaWithAthAndStdp(
        th=V_TH, w=G_W*w, g_x=G_X, t_x=T_X, rp=RP,
        stdp_params=None, wta_dist=2, wta_factor=1)
    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    # multiple sequence replay
    # include enough steps to present all triggers, let hyperexcitability
    # wear off, and then some
    len_single_run = TRIGGER_END + len(SEQ_REF) + T_X + 5
    n_steps = 2 * (len_single_run)

    # build stimulus
    drives = np.zeros((n_steps, len(nodes)))

    for start, seq_2 in zip([1, len_single_run], [SEQ_NONOVERLAP, SEQ_OVERLAP]):

        # reference sequence
        drives += 2 * make_drive_seq(
            seq=SEQ_REF, nodes=nodes, start=start, shape=drives.shape)

        # nonoverlapping sequence
        offset = len(SEQ_REF) + 2
        drives += 2 * make_drive_seq(
            seq=seq_2, nodes=nodes, start=start+offset, shape=drives.shape)

        # triggers
        trigger_seq = [SEQ_REF[0], seq_2[0]]

        for t_ctr, t in enumerate(
                start + np.arange(TRIGGER_START, TRIGGER_END, TRIGGER_INTERVAL)):
            drives[t, nodes.index(trigger_seq[t_ctr % 2])] = 2

    # run network
    rs, _ = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives)

    drive_times, drive_idxs = drives.nonzero()
    spike_times, spike_idxs = rs.nonzero()

    # plot results
    axs = [fig.add_subplot(gs[ctr, :]) for ctr in range(2)]
    axs[0].scatter(drive_times, drive_idxs, lw=0, color='b')
    axs[1].scatter(spike_times, spike_idxs, lw=0, color='k')

    axs[0].set_title('stimulus (multiple replay)')
    axs[1].set_title('network response (multiple replay)')
    axs[1].set_xlabel('time (s)')

    for ax in axs:
        ax.set_xlim(-1, 1.1 * max(drive_times.max(), spike_times.max()))
        ax.set_ylabel('node')
        plot.set_fontsize(ax, 14)

    # reverse replay
    w, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE)
    nodes, idxs = reorder_idxs(nodes, NODE_ORDER_REVERSE)
    w = w[idxs, :][:, idxs]
    ntwk = network.LocalWtaWithAthAndStdp(
        th=V_TH, w=G_W*w, g_x=G_X, t_x=T_X, rp=RP,
        stdp_params=None, wta_dist=2, wta_factor=1)
    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    len_single_run = 2 * T_X

    # build stimulus
    assert len(SEQ_REVS) == 2
    drives = np.zeros((2*len_single_run, len(nodes)))
    starts = 1 + np.arange(0, 2*len_single_run, len_single_run)

    for start, seq in zip(starts, SEQ_REVS):
        drives += 2*make_drive_seq(
            nodes=nodes, seq=seq, shape=drives.shape, start=start)

        # add forward trigger
        drives[start + len(seq) + 5, nodes.index(seq[0])] = 2
        # add reverse trigger
        drives[start + 2*(len(seq) + 5), nodes.index(seq[-1])] = 2

    # run network
    rs, _ = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives)

    drive_times, drive_idxs = drives.nonzero()
    spike_times, spike_idxs = rs.nonzero()

    # plot results
    axs = [fig.add_subplot(gs[ctr, :]) for ctr in range(2, 4)]
    nodes_reordered = reorder_idxs(nodes, NODE_ORDER_REVERSE)[1]
    axs[0].scatter(drive_times, nodes_reordered[drive_idxs], lw=0, color='b')
    axs[1].scatter(spike_times, nodes_reordered[spike_idxs], lw=0, color='k')

    axs[0].set_title('stimulus (reverse replay)')
    axs[1].set_title('network response (reverse replay)')
    axs[1].set_xlabel('time (s)')

    for ax in axs:
        ax.set_xlim(-1, 1.1 * max(drive_times.max(), spike_times.max()))
        ax.set_ylabel('node')
        plot.set_fontsize(ax, 14)

    # LIF implementation
    np.random.seed(SEED_LIF)

    # build weight matrices
    w_base, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE_LIF)
    # reorder nodes for nicer graphical presentation
    nodes, idxs = reorder_idxs(nodes, NODE_ORDER_REVERSE)
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
    axs = [fig.add_subplot(gs[ctr:ctr+2, :]) for ctr in [4, 6]]
    axs.append(fig.add_subplot(gs[8:, :]))
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


def record_extension_by_spontaneous_replay(
        SEED, GROUP_NAME, LOG_FILE,
        NETWORK_SIZE, V_TH, RP, T_X,
        NODE_SEQ, DRIVE_AMP, PROBE_TIME,
        ALPHA, G_XS, G_WS, NOISE_STDS,
        N_TRIALS, LOW_PROB_THRESHOLD, LOW_PROB_MIN_TRIALS,):
    """
    Perform a parameter sweep over varying influences of the hyperexcitability
    and the connection weight term and save the results to the database.
    """

    # preliminaries
    np.random.seed(SEED)
    session = db.connect_and_make_session('nothing_but_reruns')
    m = _models.SpontaneousReplayExtensionResult
    db.delete_record_group(session, m.group, GROUP_NAME)

    db.prepare_logging(LOG_FILE)

    # make weight matrix
    w_base, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE)

    # make pre-noise drives
    l = len(NODE_SEQ)
    drives_base = np.zeros((2*l + PROBE_TIME, len(nodes)))

    for ctr, node in enumerate(NODE_SEQ):
        drives_base[ctr + 1, nodes.index(node)] = DRIVE_AMP

    drives_base[PROBE_TIME + 1, nodes.index(NODE_SEQ[0])] = DRIVE_AMP

    node_seq_logical = (drives_base[1:1+l] > 0).astype(int)

    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    for g_x in G_XS:

        logging.info('Starting sweep set with g_x = {0:.3f}.'.format(g_x))

        # the max noise we'll consider is one in which there is a
        # 20% chance that a hyperexcitable node activates spontaneously
        noise_std_max = (1/stats.norm.ppf(0.8)) * (V_TH - g_x)
        noise_stds = [ns for ns in NOISE_STDS if 0 <= ns < noise_std_max]

        # make sure we actually have a nonzero noise to test
        if not noise_stds: continue

        # loop over all possible values of g_w
        g_ws = [g_w for g_w in G_WS if V_TH-g_x <= g_w < V_TH]
        for g_w in g_ws:

            logging.info(
                'Starting sweep with g_x = {0:.3f}, g_w = {1:.3f}.'.format(g_x, g_w))
            logging.info('Sweeping over {} noise levels...'.format(len(noise_stds)))

            ntwk = network.LocalWtaWithAthAndStdp(
                th=V_TH, w=g_w*w_base, g_x=g_x, t_x=T_X, rp=RP,
                stdp_params=None, wta_dist=2, wta_factor=ALPHA)

            # set up our data structure
            sper = _models.SpontaneousReplayExtensionResult(
                group=GROUP_NAME,
                network_size=NETWORK_SIZE,
                v_th=V_TH, rp=RP, t_x=T_X,
                sequence=NODE_SEQ,
                drive_amplitude=DRIVE_AMP,
                probe_time=PROBE_TIME,
                n_trials_attempted=N_TRIALS,
                low_probability_threshold=LOW_PROB_THRESHOLD,
                low_probability_min_trials=LOW_PROB_MIN_TRIALS,

                alpha=ALPHA,
                g_x=g_x,
                g_w=g_w,
                noise_stds=noise_stds,
                probed_replay_probs=[],
                n_trials_completed=[])

            for ns_ctr, noise_std in enumerate(noise_stds):

                broken = False

                replay_successes = []
                for tr_ctr in range(N_TRIALS):

                    # make noisy drives and run network
                    drives = drives_base + noise_std * \
                        np.random.randn(*drives_base.shape)
                    rs = ntwk.run(r_0, xc_0, drives)[0].astype(int)

                    # compare initial and probed replay sequence to true sequence
                    rs_initial = rs[1:1+l]
                    rs_replay = rs[PROBE_TIME+1:PROBE_TIME+1+l]
                    initial_match = np.all(rs_initial == node_seq_logical)
                    replay_match = np.all(rs_replay == node_seq_logical)

                    replay_successes.append(initial_match and replay_match)

                    # skip remaining trials if estimated probability is small
                    if tr_ctr + 1 >= LOW_PROB_MIN_TRIALS:
                        if np.mean(replay_successes) < LOW_PROB_THRESHOLD:
                            broken = True
                            break

                replay_prob = np.mean(replay_successes) if not broken else -1
                sper.probed_replay_probs.append(replay_prob)

                sper.n_trials_completed.append(tr_ctr + 1)

                if (ns_ctr + 1) % 5 == 0:
                    logging.info('{} noise levels completed.'.format(ns_ctr + 1))

            session.add(sper)
            session.commit()

        logging.info('All sweeps completed.')
    session.close()


def extension_by_spontaneous_replay(
        SEEDS_EXAMPLE, NOISES_EXAMPLE,
        G_W, G_X, GROUP_NAME, G_XS, X_LIM, Y_LIM):
    """
    Make plots that explore the dependence of extended replay on
    spontaneous noise level.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    m = _models.SpontaneousReplayExtensionResult

    gs = gridspec.GridSpec(2 + len(NOISES_EXAMPLE), len(G_XS))
    fig_size = (15, 3.6 * (2 + len(NOISES_EXAMPLE)))

    fig = plt.figure(figsize=fig_size, tight_layout=True)
    axs = []

    # plot statistics

    max_prob = np.max([
        np.max(srer.probed_replay_probs)
        for srer in session.query(m).filter(m.group == GROUP_NAME).all()
    ])

    for ctr, g_x in enumerate(G_XS):

        srers = session.query(m).filter(
            m.group == GROUP_NAME,
            m.g_x.between(0.999*g_x, 1.001*g_x)).order_by(m.g_w).all()

        axs.append(fig.add_subplot(gs[-1, ctr]))

        results = np.array([srer.probed_replay_probs for srer in srers]).T

        g_ws = [srer.g_w for srer in srers]
        noise_stds = srers[0].noise_stds
        d_gw = g_ws[1] - g_ws[0]
        d_noise_std = noise_stds[1] - noise_stds[0]
        extent = [
            g_ws[0] - d_gw/2, g_ws[-1] + d_gw/2,
            noise_stds[0] - d_noise_std/2, noise_stds[-1] + d_noise_std/2
        ]

        axs[-1].imshow(
            results, interpolation='nearest',
            extent=extent, origin='lower', vmin=0, vmax=max_prob, zorder=0)

        grays = results.copy()
        grays[results >=0] = np.nan
        grays[results < -0] = 0
        axs[-1].imshow(
            grays, interpolation='nearest',
            extent=extent, origin='lower', vmin=-1, vmax=1, cmap='Greys', zorder=1)

        axs[-1].set_title(
            'Replay prob. for\ng_x = {0:.2f}\n(max prob = {1:.3f})'.format(
            g_x, results.max()))

    for ax in axs:
        ax.set_xlim(X_LIM)
        ax.set_ylim(Y_LIM)
        ax.set_aspect('auto')
        ax.set_xlabel('g_w')
    axs[0].set_ylabel('noise std')

    # get parameters to use for example from last database entry
    d = srer.network_size
    v_th = srer.v_th
    rp = srer.rp
    t_x = srer.t_x
    node_seq = [tuple(pair) for pair in srer.sequence]
    drive_amp = srer.drive_amplitude
    probe_time = srer.probe_time
    alpha = srer.alpha

    # run examples
    w_base, nodes = connectivity.hexagonal_lattice(d)

    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    # run examples
    ax_drive = fig.add_subplot(gs[0, :])
    axs_example = [
        fig.add_subplot(gs[1+ctr, :], sharex=ax_drive, sharey=ax_drive)
        for ctr in range(len(NOISES_EXAMPLE))
    ]

    # make network
    ntwk = network.LocalWtaWithAthAndStdp(
        th=v_th, w=G_W*w_base, g_x=G_X, t_x=t_x, rp=rp,
        stdp_params=None, wta_dist=2, wta_factor=alpha)

    # make base drives for examples (which we'll add noise to)
    n_steps = probe_time + len(node_seq) + 2
    drives_base = np.zeros((n_steps, len(nodes)))

    # initial sequence
    for ctr, node in enumerate(node_seq):
        node_idx = nodes.index(node)
        drives_base[ctr + 1, node_idx] = drive_amp

    # replay trigger
    drives_base[probe_time, nodes.index(node_seq[0])] = drive_amp

    # show driving stimulus
    d_times, d_nodes = drives_base.nonzero()

    ax_drive.scatter(d_times, d_nodes, s=25, lw=0)

    ax_drive.axvspan(0.5, len(node_seq) + 0.5, color='r', alpha=0.15)
    ax_drive.axvspan(probe_time - 0.5, probe_time + 0.5, color='r', alpha=0.15)

    ax_drive.set_xlabel('time step')
    ax_drive.set_ylabel('node')
    ax_drive.set_title('stimulus')

    # run network
    for seed, noise, ax in zip(SEEDS_EXAMPLE, NOISES_EXAMPLE, axs_example):

        np.random.seed(seed)
        drives_example = drives_base + noise * np.random.randn(*drives_base.shape)
        rs, xcs = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives_example)

        r_times, r_nodes = rs.nonzero()

        ax.scatter(r_times, r_nodes, s=25, lw=0)

        ax.axvspan(0.5, len(node_seq) + 0.5, color='r', alpha=0.15)
        ax.axvspan(probe_time - 0.5, probe_time + 0.5, color='r', alpha=0.15)

        ax.set_xlim(-probe_time*0.05, probe_time*1.05)
        ax.set_ylabel('node')
        ax.set_title(
            'g_x = {0:.3}, g_w = {1:.3}, noise std = {2:.3}'.format(G_X, G_W, noise))

        ax.set_xlabel('time step')

    for ax in axs + [ax_drive] + axs_example: plot.set_fontsize(ax, 16)

    session.close()
    return fig


def record_replay_plus_stdp(
        SEED, GROUP, LOG_FILE,
        NETWORK_SIZE, V_TH, RP,
        SEQS_STRONG, SEQ_NOVEL, DRIVE_AMP,
        ALPHAS, BETA_0S, BETA_1S,
        T_XS, G_XS, W_0S, W_1S, NOISE_STDS,
        TRIGGER_INTERVALS, ZIP, CPRODUCT,
        TRIGGER_SEQ, INTERRUPTION_SEQ, INTERRUPTION_TIME,
        N_TRIALS, W_MEASUREMENT_TIME):
    """
    Record results of replay plus stdp.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    db.delete_record_group(session, _models.ReplayPlusStdpResult.group, GROUP)
    db.prepare_logging(LOG_FILE)
    np.random.seed(SEED)

    # make base weight matrix
    w_base, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE)

    # make mask for strong connections
    mask_w_strong = np.zeros(w_base.shape, dtype=bool)
    for seq in SEQS_STRONG:
        for node_from, node_to in zip(seq[:-1], seq[1:]):
            mask_w_strong[nodes.index(node_to), nodes.index(node_from)] = True
            mask_w_strong[nodes.index(node_from), nodes.index(node_to)] = True

    # make target masks
    mask_w_targ_for = mask_w_strong.copy()
    mask_w_targ_bi = mask_w_strong.copy()
    for node_from, node_to in zip(SEQ_NOVEL[:-1], SEQ_NOVEL[1:]):
        mask_w_targ_for[nodes.index(node_to), nodes.index(node_from)] = True
        mask_w_targ_bi[nodes.index(node_to), nodes.index(node_from)] = True
        mask_w_targ_bi[nodes.index(node_from), nodes.index(node_to)] = True

    # make pre-noise drives
    drives_base = np.zeros((1 + W_MEASUREMENT_TIME, len(nodes)))

    # initial sequence
    for ctr, node in enumerate(SEQ_NOVEL):
        node_idx = nodes.index(node)
        drives_base[ctr + 1, node_idx] = DRIVE_AMP

    # interruption sequence
    interruption_epoch = []
    if INTERRUPTION_TIME:
        for ctr, node in enumerate(INTERRUPTION_SEQ):
            t = ctr + INTERRUPTION_TIME
            node_idx = nodes.index(node)
            drives_base[t, node_idx] = DRIVE_AMP
            interruption_epoch.append(t)

    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    # loop over desired param combinations
    order = [
        'ALPHAS', 'BETA_0S', 'BETA_1S', 'T_XS', 'G_XS', 'W_0S', 'W_1S',
        'NOISE_STDS', 'TRIGGER_INTERVALS'
    ]
    parameters = zip_cproduct(ZIP, CPRODUCT, order=order, kwargs=locals())

    logging.info(
        'Beginning loop over {} parameter combinations.'.format(len(parameters)))

    for alpha, beta_0, beta_1, t_x, g_x, w_0, w_1, noise_std, trigger_interval \
            in parameters:

        # make sure variables have been assigned correctly
        assert alpha in ALPHAS and beta_0 in BETA_0S and beta_1 in BETA_1S
        assert t_x in T_XS and g_x in G_XS and w_0 in W_0S and w_1 in W_1S
        assert noise_std in NOISE_STDS and trigger_interval in TRIGGER_INTERVALS

        logging.info('Running {} trials for parameters: {}'.format(
            N_TRIALS, {
                'alpha': alpha, 'beta_0': beta_0, 'beta_1': beta_1,
                't_x': t_x, 'g_x': g_x, 'w_0': w_0, 'w_1': w_1,
                'noise_std': noise_std, 'trigger_interval': trigger_interval
            }))

        # make initial weight matrix
        w = w_0 * w_base
        w[mask_w_strong] = w_1

        # make target weight matrices
        w_targ_for = w_0 * w_base
        w_targ_bi = w_0 * w_base
        w_targ_for[mask_w_targ_for] = w_1
        w_targ_bi[mask_w_targ_bi] = w_1

        # make measurement function
        def measure_w(w_):
            dist_for = np.mean((w_ - w_targ_for) ** 2)
            dist_bi = np.mean((w_ - w_targ_bi) ** 2)
            return [dist_for, dist_bi]

        # set stdp params
        stdp_params = {
            'w_0': w_0, 'w_1': w_1, 'beta_0': beta_0, 'beta_1': beta_1
        }

        # make network
        ntwk = network.LocalWtaWithAthAndStdp(
            th=V_TH, w=w, g_x=g_x, t_x=t_x, rp=RP,
            stdp_params=stdp_params, wta_dist=2, wta_factor=alpha)

        # add triggers to base drives
        drives = drives_base.copy()

        if trigger_interval:
            trigger_times = np.arange(
                1, 1 + W_MEASUREMENT_TIME, trigger_interval)[1:]

            for ctr, trigger_time in enumerate(trigger_times):
                if trigger_time in interruption_epoch: continue

                node = TRIGGER_SEQ[ctr % len(TRIGGER_SEQ)]
                node_idx = nodes.index(node)
                drives[trigger_time, node_idx] = DRIVE_AMP

        # create data structure
        rpsr = _models.ReplayPlusStdpResult(
            group=GROUP,
            network_size=NETWORK_SIZE,
            v_th=V_TH,
            rp=RP,
            sequences_strong=SEQS_STRONG,
            sequence_novel=SEQ_NOVEL,
            drive_amplitude=DRIVE_AMP,

            alpha=alpha,
            t_x=t_x,
            g_x=g_x,
            w_0=w_0,
            w_1=w_1,
            noise_std=noise_std,
            beta_0=beta_0,
            beta_1=beta_1,

            trigger_interval=trigger_interval,
            trigger_sequence=TRIGGER_SEQ,
            interruption_time=INTERRUPTION_TIME,
            interruption_sequence=INTERRUPTION_SEQ,

            w_measurement_time=W_MEASUREMENT_TIME,

            n_trials_completed=0,
            w_scores=[])

        # loop over trials
        for tr_ctr in range(N_TRIALS):

            # add noise
            drives_ = drives + (noise_std * np.random.randn(*drives.shape))

            # run network
            rs, _, w_measurements = ntwk.run(
                r_0, xc_0, drives_, measure_w=measure_w)

            rpsr.w_scores.append(w_measurements[-1])
            rpsr.n_trials_completed += 1

            if (tr_ctr + 1) % 25 == 0:
                logging.info('{} trials completed.'.format(tr_ctr + 1))

        session.add(rpsr)
        session.commit()
    session.close()


def _replay_plus_stdp_example(
        axs, label, seed, network_size, v_th, rp, alpha, beta_0, beta_1,
        t_x, g_x, w_0, w_1, seqs_strong, seq_novel, drive_amp,
        duration, noise_std, trigger_interval, trigger_seq,
        interruption_time, interruption_seq, node_order):
    """
    Run a single simulation and plot the results on a set of three axes.
    """
    # make base weight matrix
    w_base, nodes = connectivity.hexagonal_lattice(network_size)
    nodes_reordered = reorder_idxs(nodes, node_order)

    # make mask for strong connections
    mask_w_strong = np.zeros(w_base.shape, dtype=bool)
    for seq in seqs_strong:
        for node_from, node_to in zip(seq[:-1], seq[1:]):
            mask_w_strong[nodes.index(node_to), nodes.index(node_from)] = True
            mask_w_strong[nodes.index(node_from), nodes.index(node_to)] = True

    # make target masks
    mask_w_targ_for = mask_w_strong.copy()
    mask_w_targ_bi = mask_w_strong.copy()
    for node_from, node_to in zip(seq_novel[:-1], seq_novel[1:]):
        mask_w_targ_for[nodes.index(node_to), nodes.index(node_from)] = True
        mask_w_targ_bi[nodes.index(node_to), nodes.index(node_from)] = True
        mask_w_targ_bi[nodes.index(node_from), nodes.index(node_to)] = True

    # make network
    w = w_0 * w_base
    w[mask_w_strong] = w_1

    # make target weight matrices
    w_targ_for = w_0 * w_base
    w_targ_bi = w_0 * w_base
    w_targ_for[mask_w_targ_for] = w_1
    w_targ_bi[mask_w_targ_bi] = w_1

    # make measurement function
    def measure_w(w_):
        dist_for = np.mean((w_ - w_targ_for) ** 2)
        dist_bi = np.mean((w_ - w_targ_bi) ** 2)
        return [dist_for, dist_bi]

    ntwk = network.LocalWtaWithAthAndStdp(
        th=v_th, w=w, g_x=g_x, t_x=t_x, rp=rp,
        stdp_params={'w_0': w_0, 'w_1': w_1, 'beta_0': beta_0, 'beta_1': beta_1},
        wta_dist=2, wta_factor=alpha)

    # make pre-noise drives
    drives_base = np.zeros((1 + duration, len(nodes)))

    # initial sequence
    for ctr, node in enumerate(seq_novel):
        node_idx = nodes.index(node)
        drives_base[ctr + 1, node_idx] = drive_amp

    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    np.random.seed(seed)
    drives = drives_base.copy()

    # add interruption sequence if present
    interruption_epoch = []
    if interruption_time:
        for n_ctr, node in enumerate(interruption_seq):
            t = n_ctr + interruption_time
            node_idx = nodes.index(node)
            drives[t, node_idx] = drive_amp
            interruption_epoch.append(t)

    # add triggers
    if trigger_interval:
        trigger_times = np.arange(1, 1 + duration, trigger_interval)[1:]

        for t_ctr, t in enumerate(trigger_times):
            if t in interruption_epoch: continue

            node = trigger_seq[t_ctr % len(trigger_seq)]
            node_idx = nodes.index(node)
            drives[t, node_idx] = drive_amp

    drive_times, drive_idxs = drives.nonzero()

    # add noise
    drives += noise_std * np.random.randn(*drives.shape)

    # run network
    rs, _, w_measurements = ntwk.run(
        r_0=r_0, xc_0=xc_0, drives=drives, measure_w=measure_w)
    w_measurements = np.array(w_measurements)
    dists_w_for = w_measurements[:, 0]
    dists_w_bi = w_measurements[:, 1]

    spike_times, spike_idxs = rs.nonzero()

    # plot results
    axs[0].scatter(drive_times, nodes_reordered[drive_idxs], s=10, lw=0)
    axs[1].scatter(spike_times, nodes_reordered[spike_idxs], s=10, lw=0)
    handle_0 = axs[2].plot(dists_w_for, color='r', lw=2, label='for')[0]
    handle_1 = axs[2].plot(dists_w_bi, color='c', lw=2, label='bi')[0]

    axs[0].set_ylim(-1, 1.5 * len(node_order))
    axs[1].set_ylim(-1, 1.5 * len(node_order))

    axs[2].set_xlim(-1, len(drives))

    axs[0].set_title('stimulus ({})'.format(label))
    axs[1].set_title('network response')
    axs[2].set_title('novel sequence weights')

    axs[2].set_xlabel('time step')

    for ax in axs[:2]: ax.set_ylabel('node')
    axs[2].set_ylabel('weight')
    axs[2].legend(handles=[handle_0, handle_1], loc='best')

    return axs


def replay_plus_stdp_periodic_stim(
        NETWORK_SIZE, V_TH, RP,
        ALPHA, BETA_0, BETA_1, T_X, G_X, W_0, W_1,
        SEQS_STRONG, SEQ_NOVEL, DRIVE_AMP, DURATION,
        LABEL, SEED, NOISE_STD, TRIGGER_INTERVAL, TRIGGER_SEQ,
        INTERRUPTION_TIME, INTERRUPTION_SEQ, NODE_ORDER,
        GROUP, G_XS_STATS):
    """
    Display example traces of network dynamics with replay and stdp.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    assert len(G_XS_STATS) <= 2
    fig = plt.figure(figsize=(15, 7), tight_layout=True)

    # run example
    gs = gridspec.GridSpec(3, 5)
    axs_ex = [fig.add_subplot(gs[0, :-2])]
    axs_ex.extend([fig.add_subplot(gs[i, :-2], sharex=axs_ex[0]) for i in range(1, 3)])

    _replay_plus_stdp_example(
        axs=axs_ex, seed=SEED, network_size=NETWORK_SIZE, v_th=V_TH, rp=RP,
        alpha=ALPHA, beta_0=BETA_0, beta_1=BETA_1, t_x=T_X, g_x=G_X,
        w_0=W_0, w_1=W_1, seqs_strong=SEQS_STRONG, seq_novel=SEQ_NOVEL,
        drive_amp=DRIVE_AMP, duration=DURATION, label=LABEL,
        noise_std=NOISE_STD, trigger_interval=TRIGGER_INTERVAL,
        trigger_seq=TRIGGER_SEQ, interruption_time=INTERRUPTION_TIME,
        interruption_seq=INTERRUPTION_SEQ, node_order=NODE_ORDER)

    # plot stats from saved simulation results
    ax = fig.add_subplot(gs[:-1, -2:])
    ax_legend = fig.add_subplot(gs[-1, -2:])
    hs = []  # legend handles

    # loop over g_x's
    for g_x, ls in zip(G_XS_STATS, ['-', '--']):
        # get all results for this g_x
        rpsrs_all = session.query(_models.ReplayPlusStdpResult).filter(
            _models.ReplayPlusStdpResult.group == GROUP,
            _models.ReplayPlusStdpResult.g_x.between(.99*g_x, 1.01*g_x))

        # get all beta_1's that were tested
        beta_1s = session.query(_models.ReplayPlusStdpResult.beta_1).filter(
            _models.ReplayPlusStdpResult.group == GROUP,
            _models.ReplayPlusStdpResult.g_x.between(.99*g_x, 1.01*g_x)
        ).order_by(_models.ReplayPlusStdpResult.beta_1).distinct().all()

        beta_1s = [i[0] for i in beta_1s]

        # get colors for forward and reverse weights
        cs_f = plot.get_n_colors(len(beta_1s), 'afmhot')
        cs_b = plot.get_n_colors(len(beta_1s), 'winter')

        # for each beta_1 plot mean final forward and reverse weights
        # vs. trigger interval
        for beta_1, c_f, c_b in zip(beta_1s, cs_f, cs_b):

            # get all results with this beta_1
            rpsrs = rpsrs_all.filter(
                _models.ReplayPlusStdpResult.beta_1.between(.99*beta_1, 1.01*beta_1)
            ).order_by(
                _models.ReplayPlusStdpResult.trigger_interval).all()

            # get trigger intervals and final measured weights
            tis = [rpsr.trigger_interval for rpsr in rpsrs]
            ws = [np.array(rpsr.w_scores) for rpsr in rpsrs]

            for fr_label, c in zip(['for', 'bi'], [c_f, c_b]):

                if fr_label == 'for':
                    w_scores = [w[:, 0] for w in ws]
                elif fr_label == 'bi':
                    w_scores = [w[:, 1] for w in ws]

                means = [w_score.mean() for w_score in w_scores]

                h = ax.plot(
                    tis, means, color=c, lw=2, ls=ls,
                    label='beta = {0:.2f} ({1})'.format(beta_1, fr_label[:3]))[0]

                if ls == '-':
                    hs.append(h)

    ax.set_xlabel('trigger interval')
    ax.set_ylabel('score')

    hs = hs[::2] + hs[1::2]
    ax_legend.legend(handles=hs, ncol=2, loc='best')
    ax_legend.get_xaxis().set_visible(False)
    ax_legend.get_yaxis().set_visible(False)

    for ax in axs_ex + [ax]: plot.set_fontsize(ax, 14)
    session.close()
    return fig


def replay_plus_stdp_spontaneous(
        NETWORK_SIZE, V_TH, RP,
        ALPHA, BETA_0, BETA_1, T_X, G_X, W_0, W_1,
        SEQS_STRONG, SEQ_NOVEL, DRIVE_AMP, DURATION,
        LABEL, SEED, NOISE_STD, TRIGGER_INTERVAL, TRIGGER_SEQ,
        INTERRUPTION_TIME, INTERRUPTION_SEQ, NODE_ORDER,
        GROUP, G_XS_STATS):
    """
    Show example plots and statistics for replay + stdp simulation
    driven by spontaneous noise.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    fig = plt.figure(figsize=(15, 7), tight_layout=True)

    # run example
    gs = gridspec.GridSpec(3, 5)
    axs_ex = [fig.add_subplot(gs[0, :-2])]
    axs_ex.extend([fig.add_subplot(gs[i, :-2], sharex=axs_ex[0]) for i in range(1, 3)])

    _replay_plus_stdp_example(
        axs=axs_ex, seed=SEED, network_size=NETWORK_SIZE, v_th=V_TH, rp=RP,
        alpha=ALPHA, beta_0=BETA_0, beta_1=BETA_1, t_x=T_X, g_x=G_X,
        w_0=W_0, w_1=W_1, seqs_strong=SEQS_STRONG, seq_novel=SEQ_NOVEL,
        drive_amp=DRIVE_AMP, duration=DURATION, label=LABEL,
        noise_std=NOISE_STD, trigger_interval=TRIGGER_INTERVAL,
        trigger_seq=TRIGGER_SEQ, interruption_time=INTERRUPTION_TIME,
        interruption_seq=INTERRUPTION_SEQ, node_order=NODE_ORDER)

    # plot stats from saved simulation results
    ax = fig.add_subplot(gs[:, -2:])
    hs = []  # legend handles

    for g_x, ls in zip(G_XS_STATS, ['-', '--']):
        rpsrs = session.query(_models.ReplayPlusStdpResult).filter(
            _models.ReplayPlusStdpResult.group == GROUP,
            _models.ReplayPlusStdpResult.g_x.between(.99*g_x, 1.01*g_x)).order_by(
            _models.ReplayPlusStdpResult.noise_std).all()

        noise_stds = [rpsr.noise_std for rpsr in rpsrs]
        ws = [np.array(rpsr.w_scores) for rpsr in rpsrs]

        for fr_label, c in zip(['for', 'bi'], ['r', 'c']):

            if fr_label == 'for':
                w_scores = [w[:, 0] for w in ws]
            elif fr_label == 'bi':
                w_scores = [w[:, 1] for w in ws]

            means = [w_score.mean() for w_score in w_scores]

            h = ax.plot(
                noise_stds, means, lw=2, color=c, ls=ls,
                label='G_X = {0:.1f} ({1})'.format(g_x, fr_label))[0]
            hs.append(h)

    ax.set_xlabel('noise std')
    ax.legend(handles=hs)

    for ax in axs_ex + [ax]: plot.set_fontsize(ax, 14)
    session.close()
    return fig


def replay_plus_stdp_interrupted(NETWORK_SIZE, V_TH, RP,
        ALPHA, BETA_0, BETA_1, T_X, G_X_EX, W_0, W_1,
        SEQS_STRONG, SEQ_NOVEL, DRIVE_AMP, DURATION,
        LABEL, SEED, NOISE_STD, TRIGGER_INTERVAL, TRIGGER_SEQ,
        INTERRUPTION_TIME, INTERRUPTION_SEQ, NODE_ORDER,
        GROUP, G_X):
    """
    Show examples and stats for replay + stdp simulation with interrupting
    strong sequence.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    fig = plt.figure(figsize=(15, 7), tight_layout=True)

    # run example
    gs = gridspec.GridSpec(3, 5)
    axs_ex = [fig.add_subplot(gs[0, :-2])]
    axs_ex.extend([fig.add_subplot(gs[i, :-2], sharex=axs_ex[0]) for i in range(1, 3)])

    _replay_plus_stdp_example(
        axs=axs_ex, seed=SEED, network_size=NETWORK_SIZE, v_th=V_TH, rp=RP,
        alpha=ALPHA, beta_0=BETA_0, beta_1=BETA_1, t_x=T_X, g_x=G_X_EX,
        w_0=W_0, w_1=W_1, seqs_strong=SEQS_STRONG, seq_novel=SEQ_NOVEL,
        drive_amp=DRIVE_AMP, duration=DURATION, label=LABEL,
        noise_std=NOISE_STD, trigger_interval=TRIGGER_INTERVAL,
        trigger_seq=TRIGGER_SEQ, interruption_time=INTERRUPTION_TIME,
        interruption_seq=INTERRUPTION_SEQ, node_order=NODE_ORDER)

    # plot stats from saved simulation results
    ax = fig.add_subplot(gs[:-1, -2:])
    ax_legend = fig.add_subplot(gs[-1, -2:])
    hs = []  # legend handles

    # get all results for this group
    rpsrs_all = session.query(_models.ReplayPlusStdpResult).filter(
        _models.ReplayPlusStdpResult.group == GROUP,
        _models.ReplayPlusStdpResult.g_x.between(.99*G_X, 1.01*G_X))

    # get all beta_1s
    beta_1s = session.query(_models.ReplayPlusStdpResult.beta_1).filter(
        _models.ReplayPlusStdpResult.group == GROUP,
        _models.ReplayPlusStdpResult.g_x.between(.99*G_X, 1.01*G_X)).order_by(
        _models.ReplayPlusStdpResult.beta_1).distinct().all()
    beta_1s = [i[0] for i in beta_1s]

    colors = plot.get_n_colors(len(beta_1s), 'afmhot')

    # loop over beta_1 and plot mean w vs. alpha for each
    for beta_1, c in zip(beta_1s, colors):
        rpsrs = rpsrs_all.filter(
            _models.ReplayPlusStdpResult.beta_1.between(.99*beta_1, 1.01*beta_1)
        ).order_by(_models.ReplayPlusStdpResult.alpha).all()

        alphas = [rpsr.alpha for rpsr in rpsrs]
        ws = [np.array(rpsr.w_scores) for rpsr in rpsrs]

        w_scores = [w[:, 0].mean() for w in ws]

        h = ax.plot(
            alphas, w_scores, color=c, lw=2,
            label='beta_1 = {0:.2f}'.format(beta_1))[0]
        hs.append(h)

    ax.set_xlabel('alpha')

    ax_legend.legend(handles=hs, ncol=2, loc='best')
    ax_legend.get_xaxis().set_visible(False)
    ax_legend.get_yaxis().set_visible(False)

    for ax in axs_ex + [ax, ax_legend]: plot.set_fontsize(ax, 14)
    session.close()
    return fig
