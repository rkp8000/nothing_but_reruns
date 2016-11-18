from __future__ import division, print_function
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import threading

import connectivity
from db import _models
import db
import network
import plot


def record_extension_by_spontaneous_replay(
        SEED, GROUP_NAME, LOG_FILE,
        NETWORK_SIZE, V_TH, RP, T_X,
        NODE_SEQ, DRIVE_AMP, PROBE_TIME,
        ALPHA, G_X_STEP, G_W_STEP, NOISE_STEP,
        N_TRIALS, LOW_PROB_THRESHOLD, LOW_PROB_MIN_TRIALS,):
    """
    Perform a parameter sweep over varying influences of the hyperexcitability
    and the connection weight term and save the results to the database.
    """

    # preliminaries
    np.random.seed(SEED)
    s = db.connect_and_make_session('neural_pattern_replay')
    m = _models.SpontaneousReplayExtensionResult
    db.delete_record_group(s, m.group, GROUP_NAME)

    db.prepare_logging(LOG_FILE)

    g_xs = np.arange(0, V_TH, G_X_STEP)[1:]

    # make weight matrix
    w_base, nodes = connectivity.hexagonal_lattice(NETWORK_SIZE)

    # make pre-noise drives
    l = len(NODE_SEQ)
    drives_base = np.zeros((2*l + PROBE_TIME, len(nodes)))

    for ctr, node in NODE_SEQ: drives_base[ctr + 1, nodes.index(node)] = DRIVE_AMP
    drives_base[PROBE_TIME + 1, nodes.index(NODE_SEQ[0])] = DRIVE_AMP

    node_seq_logical = drives_base[1:1+l, :]

    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    for g_x in g_xs:

        logging.info('Starting sweep set with g_x = {0:.3f}.'.format(g_x))

        # the max noise we'll consider is one in which there is a
        # 20% chance that a hyperexcitable node activates spontaneously
        noise_std_max = (1/stats.norm.ppf(0.8)) * (v_th - g_x)
        noise_stds = np.arange(0, noise_std_max, NOISE_STEP)

        # make sure we actually have a nonzero noise to test
        if len(noise_stds) > 1: noise_stds = noise_stds[1:]
        else: continue

        # loop over all possible values of g_w
        g_ws = np.arange(V_TH-g_x, V_TH, G_W_STEP)
        for g_w in g_ws:

            logging.info('Starting sweep with g_x = {0:.3f}, g_w = {1:.3f}.'.format(g_x, g_w))
            logging.info('Sweeping over {} noise levels...'.format(len(noise_stds)))

            ntwk = network.LocalWtaWithAthAndStdp(
                th=v_th, w=g_w*w_base, g_x=g_x, t_x=t_x, rp=rp,
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

            for noise_std in noise_stds:

                replay_successes = []
                for tr_ctr in range(N_TRIALS):

                    # make noisy drives and run network
                    drives = drives_base + noise_std * np.random.randn(*drives_base.shape)
                    rs = ntwk.run(r_0, xc_0, drives)[0]

                    # compare initial and probed replay sequence to true sequence
                    initial_match = np.all(rs.T[1:1+l] == node_seq_logical)
                    replay_match = np.all(rs.T[PROBE_TIME+1:PROBE_TIME+1+l] == node_seq_logical)

                    replay_successes.append(initial_match and replay_match)

                    # skip remaining trials if estimated probability is sufficiently small
                    if tr_ctr + 1 >= LOW_PROB_MIN_TRIALS:
                        if np.mean(replay_successes) < LOW_PROB_THRESHOLD:
                            break

                sper.probed_replay_probs.append(np.mean(replay_successes))
                sper.n_trials_completed.append(tr_ctr + 1)

            session.add(sper)
            session.commit()

        logging.info('All sweeps completed.')
    session.close()


def extension_by_spontaneous_replay(
        SEEDS_EXAMPLE, NOISES_EXAMPLE,
        SEED_STATS, NOISES_STATS, N_TRIALS,
        D, W, TH, G_X, T_X, RP, WS_STATS, G_XS_STATS,
        NODE_SEQ, REPLAY_TRIGGER, DRIVE_AMP):
    """
    Make plots that explore the dependence of extended replay on
    spontaneous noise level.
    """
    w_base, nodes = connectivity.hexagonal_lattice(D)

    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))

    # set up figure layout
    gs = gridspec.GridSpec(len(SEEDS_EXAMPLE), 3)
    fig = plt.figure(figsize=(15, 2 * len(SEEDS_EXAMPLE)), tight_layout=True)

    # run examples
    axs_example = [fig.add_subplot(gs[0, :2])]
    axs_example.extend([
        fig.add_subplot(gs[1+ctr, :2], sharex=axs_example[0], sharey=axs_example[0])
        for ctr in range(len(SEEDS_EXAMPLE) - 1)])

    # make network
    ntwk = network.LocalWtaWithAthAndStdp(
        th=TH, w=W*w_base, g_x=G_X, t_x=T_X, rp=RP,
        stdp_params=None, wta_dist=2)

    # make base drives for examples (which we'll add noise to)
    n_steps = REPLAY_TRIGGER + len(NODE_SEQ) + 2
    drives_base = np.zeros((n_steps, len(nodes)))

    # initial sequence
    for ctr, node in enumerate(NODE_SEQ):
        node_idx = nodes.index(node)
        drives_base[ctr + 1, node_idx] = DRIVE_AMP

    # replay trigger
    drives_base[REPLAY_TRIGGER, nodes.index(NODE_SEQ[0])] = DRIVE_AMP

    # run network
    for seed, noise, ax in zip(SEEDS_EXAMPLE, NOISES_EXAMPLE, axs_example):

        np.random.seed(seed)
        drives_example = drives_base + noise * np.random.randn(*drives_base.shape)
        rs, xcs = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives_example)

        r_times, r_nodes = rs.nonzero()

        ax.scatter(r_times, r_nodes, s=15, lw=0)
        ax.set_xlim(-REPLAY_TRIGGER*0.05, REPLAY_TRIGGER*1.05)
        ax.set_ylabel('node')
        ax.set_title('noise stdev = {0:.3}'.format(noise))

    axs_example[-1].set_xlabel('time step')

    # run statistics
    ax_stats = fig.add_subplot(gs[:, -1])
    colors = plot.get_n_colors(len(WS_STATS), 'jet')
    handles = []

    for w, g_x, color in zip(WS_STATS, G_XS_STATS, colors):

        label = 'w = {0:.1f}, g_x = {1:.1f}'.format(w, g_x)
        ntwk = network.LocalWtaWithAthAndStdp(
            th=TH, w=w*w_base, g_x=g_x, t_x=T_X, rp=RP,
            stdp_params=None, wta_dist=2)

        np.random.seed(SEED_STATS)
        replay_probs = []

        for noise in NOISES_STATS:
            correct_replays = 0

            for _ in range(N_TRIALS):

                # create new instance of noisy drives and run network
                drives = drives_base + noise * np.random.randn(*drives_base.shape)
                rs, _ = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives)

                # compare triggered sequence to initial sequence
                rs_initial = rs[1:1+len(NODE_SEQ)]
                rs_replay = rs[REPLAY_TRIGGER:REPLAY_TRIGGER+len(NODE_SEQ)]

                if np.all(rs_initial == rs_replay): correct_replays += 1

            replay_probs.append(correct_replays / N_TRIALS)

        handles.append(ax_stats.plot(replay_probs, NOISES_STATS, lw=2, label=label)[0])

    ax_stats.invert_yaxis()
    ax_stats.set_ylabel('noise level')
    ax_stats.set_xlabel('correct replay probability\n(at t = {})'.format(REPLAY_TRIGGER))
    ax_stats.legend(handles=handles, loc='best')

    for ax in axs_example + [ax_stats]: plot.set_fontsize(ax, 16)

    return fig
