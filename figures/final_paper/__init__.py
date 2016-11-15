from __future__ import division, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import connectivity
import network
import plot


def extended_replay_noise_dependence(
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
