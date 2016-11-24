from __future__ import division, print_function
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import connectivity
from db import _models
import db
import network
import plot


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

            logging.info('Starting sweep with g_x = {0:.3f}, g_w = {1:.3f}.'.format(g_x, g_w))
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
                    drives = drives_base + noise_std * np.random.randn(*drives_base.shape)
                    rs = ntwk.run(r_0, xc_0, drives)[0].astype(int)

                    # compare initial and probed replay sequence to true sequence
                    rs_initial = rs[1:1+l]
                    rs_replay = rs[PROBE_TIME+1:PROBE_TIME+1+l]
                    initial_match = np.all(rs_initial == node_seq_logical)
                    replay_match = np.all(rs_replay == node_seq_logical)

                    replay_successes.append(initial_match and replay_match)

                    # skip remaining trials if estimated probability is sufficiently small
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
        W, G_X, GROUP_NAME, G_XS, X_LIM, Y_LIM):
    """
    Make plots that explore the dependence of extended replay on
    spontaneous noise level.
    """
    # preliminaries
    session = db.connect_and_make_session('nothing_but_reruns')
    m = _models.SpontaneousReplayExtensionResult

    gs = gridspec.GridSpec(1 + len(NOISES_EXAMPLE), len(G_XS))
    fig_size = (15, 4 * (1 + len(NOISES_EXAMPLE)))

    fig = plt.figure(figsize=fig_size, tight_layout=True)
    axs = []

    # plot statistics

    max_prob = np.max([
        np.max(srer.probed_replay_probs)
        for srer in session.query(m).filter(m.group == GROUP_NAME).all()
    ])

    print('Max replay prob = {}'.format(max_prob))

    for ctr, g_x in enumerate(G_XS):

        srers = session.query(m).filter(
            m.group == GROUP_NAME,
            m.g_x.between(0.999*g_x, 1.001*g_x)).order_by(m.g_w).all()

        axs.append(fig.add_subplot(gs[0, ctr]))

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

        axs[-1].set_title('g_x = {0:.2f}'.format(g_x))

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
    axs_example = [fig.add_subplot(gs[1, :])]
    axs_example.extend([
        fig.add_subplot(gs[2+ctr, :], sharex=axs_example[0], sharey=axs_example[0])
        for ctr in range(len(SEEDS_EXAMPLE) - 1)])

    # make network
    ntwk = network.LocalWtaWithAthAndStdp(
        th=v_th, w=W*w_base, g_x=G_X, t_x=t_x, rp=rp,
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

    # run network
    for seed, noise, ax in zip(SEEDS_EXAMPLE, NOISES_EXAMPLE, axs_example):

        np.random.seed(seed)
        drives_example = drives_base + noise * np.random.randn(*drives_base.shape)
        rs, xcs = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives_example)

        r_times, r_nodes = rs.nonzero()

        ax.scatter(r_times, r_nodes, s=15, lw=0)
        ax.set_xlim(-probe_time*0.05, probe_time*1.05)
        ax.set_ylabel('node')
        ax.set_title('noise stdev = {0:.3}'.format(noise))

    axs_example[-1].set_xlabel('time step')

    for ax in axs + axs_example: plot.set_fontsize(ax, 16)

    session.close()
    return fig
