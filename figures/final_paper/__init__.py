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
from shortcuts import zip_cproduct


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

    # measurement function
    ws_to_measure = [
        (nodes.index(node_to), nodes.index(node_from))
        for node_from, node_to in zip(SEQ_NOVEL[:-1], SEQ_NOVEL[1:])
    ] + [
        (nodes.index(node_to), nodes.index(node_from))
        for node_from, node_to in zip(SEQ_NOVEL[::-1][:-1], SEQ_NOVEL[::-1][1:])
    ]

    def measure_w(w):
        return [w[w_to_measure] for w_to_measure in ws_to_measure]

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

        # make weight matrix
        w = w_0 * w_base
        w[mask_w_strong] = w_1

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
            ws_measured=ws_to_measure,

            n_trials_completed=0,
            ws_measured_values=[])

        # loop over trials
        for _ in range(N_TRIALS):
            # add noise
            drives += noise_std * np.random.randn(*drives.shape)

            # run network
            rs, _, w_measurements = ntwk.run(
                r_0, xc_0, drives, measure_w=measure_w)

            rpsr.ws_measured_values.append(w_measurements[-1])
            rpsr.n_trials_completed += 1

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
    nodes_reordered = plot.reorder_idxs(nodes, node_order)

    # make mask for strong connections
    mask_w_strong = np.zeros(w_base.shape, dtype=bool)
    for seq in seqs_strong:
        for node_from, node_to in zip(seq[:-1], seq[1:]):
            mask_w_strong[nodes.index(node_to), nodes.index(node_from)] = True
            mask_w_strong[nodes.index(node_from), nodes.index(node_to)] = True

    # make network
    w = w_0 * w_base
    w[mask_w_strong] = w_1

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

    # measurement function
    ws_to_measure = [
        (nodes.index(node_to), nodes.index(node_from))
        for node_from, node_to in zip(seq_novel[:-1], seq_novel[1:])
    ] + [
        (nodes.index(node_to), nodes.index(node_from))
        for node_from, node_to in zip(seq_novel[::-1][:-1], seq_novel[::-1][1:])
    ]

    def measure_w(w):
        return [w[w_to_measure] for w_to_measure in ws_to_measure]

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
    n_ws = w_measurements.shape[1]

    spike_times, spike_idxs = rs.nonzero()
    w_forwards = w_measurements[:, :int(n_ws/2)].mean(axis=1)
    w_reverses = w_measurements[:, int(n_ws/2):].mean(axis=1)

    # plot results
    axs[0].scatter(drive_times, nodes_reordered[drive_idxs], s=10, lw=0)
    axs[1].scatter(spike_times, nodes_reordered[spike_idxs], s=10, lw=0)
    handle_0 = axs[2].plot(w_forwards, color='k', lw=2, label='forward')[0]
    handle_1 = axs[2].plot(w_reverses, color='c', lw=2, label='reverse')[0]

    axs[0].set_ylim(-1, 1.5 * len(node_order))
    axs[1].set_ylim(-1, 1.5 * len(node_order))

    axs[2].set_xlim(-1, len(drives))
    axs[2].set_ylim(w_0, w_1)

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
    assert len(G_XS_STATS) == 2
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
        cs_r = plot.get_n_colors(len(beta_1s), 'winter')

        # for each beta_1 plot mean final forward and reverse weights
        # vs. trigger interval
        for beta_1, c_f, c_r in zip(beta_1s, cs_f, cs_r):

            # get all results with this beta_1
            rpsrs = rpsrs_all.filter(
                _models.ReplayPlusStdpResult.beta_1.between(.99*beta_1, 1.01*beta_1)
            ).order_by(
                _models.ReplayPlusStdpResult.trigger_interval).all()

            # get trigger intervals and final measured weights
            tis = [rpsr.trigger_interval for rpsr in rpsrs]
            ws_all = [np.array(rpsr.ws_measured_values) for rpsr in rpsrs]
            n_weights = int(ws_all[0].shape[1])

            for fr_label, c in zip(['forward', 'reverse'], [c_f, c_r]):

                if fr_label == 'forward':
                    ws = [w[0, :int(n_weights/2)] for w in ws_all]
                elif fr_label == 'reverse':
                    ws = [w[0, int(n_weights/2):] for w in ws_all]

                means = [w.mean() for w in ws]
                h = ax.plot(
                    tis, means, color=c, lw=2, ls=ls,
                    label='beta = {0:.2f} ({1})'.format(beta_1, fr_label[:3]))[0]

                if ls == '-':
                    hs.append(h)

    ax.axhline(rpsr.w_0, color='k', zorder=-1)
    ax.axhline(rpsr.w_1, color='k', zorder=-1)

    w_range = rpsr.w_1 - rpsr.w_0
    ax.set_ylim(rpsr.w_0 - .1*w_range, rpsr.w_1 + .1*w_range)

    ax.set_xlabel('trigger interval')
    ax.set_ylabel('E[W(t=400)]')

    hs = hs[::2] + hs[1::2]
    ax_legend.legend(handles=hs, ncol=2, loc='best')
    ax_legend.get_xaxis().set_visible(False)
    ax_legend.get_yaxis().set_visible(False)

    for ax in axs_ex + [ax]: plot.set_fontsize(ax, 14)
    session.close()
    return fig
