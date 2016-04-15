from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from connectivity import feed_forward_grid
from network import SoftmaxWTAWithLingeringHyperexcitability as network
from plot import fancy_raster, set_fontsize


def make_feed_forward_grid_weights(shape, spread):

    w = feed_forward_grid(shape=shape, spread=1)
    w += feed_forward_grid(shape=shape, spread=spread)

    return w


def replay_example(
        SEED, GRID_SHAPE, LATERAL_SPREAD, G_W, G_X, G_D, T_X,
        DRIVEN_NODES, DRIVE_AMPLITUDE,
        AX_SIZE, FONT_SIZE):

    # RUN SIMULATION

    np.random.seed(SEED)
    n_trials = len(DRIVEN_NODES)

    # make network

    w = make_feed_forward_grid_weights(GRID_SHAPE, LATERAL_SPREAD)
    ntwk = network(w=w, g_w=G_W, g_x=G_X, g_d=G_D, t_x=T_X)

    # loop over trials

    all_drives = []
    all_rs = []

    for driven_nodes_one_trial in DRIVEN_NODES:

        run_time = 2 * len(driven_nodes_one_trial)
        n_nodes = w.shape[0]

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


    # MAKE PLOTS

    fig, axs = plt.subplots(
        n_trials, 1, figsize=(AX_SIZE[0], AX_SIZE[1] * n_trials), facecolor='white',
        sharex=True, sharey=True, tight_layout=True)

    if n_trials == 1:

        axs = [axs]

    for ax, drives, rs in zip(axs, all_drives, all_rs):

        fancy_raster(ax, rs, drives)

    for ax_ctr, ax in enumerate(axs):

        ax.set_ylim(-1, n_nodes)
        ax.set_ylabel('ensemble')

        if ax_ctr == n_trials - 1:
            ax.set_xlabel('time step')

        set_fontsize(ax, FONT_SIZE)


def replay_probability_calculation(
        SEED, GRID_SHAPE, LATERAL_SPREAD,
        G_BARS, G_W_STARS, G_X_STARS, G_D_STARS, T_X,
        TEST_SEQUENCES, DRIVEN_NODES, DRIVE_AMPLITUDE,
        AX_SIZE, FONT_SIZE):

    # RUN SIMULATIONS

    np.random.seed(SEED)

    # do some preliminary computations

    n_seqs = len(TEST_SEQUENCES)
    run_times = [len(seq) for seq in TEST_SEQUENCES]
    fixed_params_all = zip(G_W_STARS, G_X_STARS, G_D_STARS)

    # make weight matrix

    w = make_feed_forward_grid_weights(GRID_SHAPE, LATERAL_SPREAD)
    n_nodes = w.shape[0]

    # loop over all sequences

    p_seqs_alls = []

    for test_sequence, run_time, driven_node in zip(TEST_SEQUENCES, run_times, DRIVEN_NODES):

        # set up initial state, initial hyperexcitability counter, and drives

        r_0 = np.zeros((n_nodes,))

        xc_0 = np.zeros((n_nodes,))
        xc_0[test_sequence] = np.arange(T_X - run_time, T_X) + 1

        drives = np.zeros((run_time, n_nodes))
        drives[0, driven_node] = DRIVE_AMPLITUDE

        # loop through all "fixed parameter sets"

        p_seqs_all = []

        for g_w_star, g_x_star, g_d_star in fixed_params_all:

            p_seqs = []

            for g_bar in G_BARS:

                g_w = g_bar * g_w_star
                g_x = g_bar * g_x_star
                g_d = g_bar * g_d_star

                ntwk = network(w=w, g_w=g_w, g_x=g_x, g_d=g_d, t_x=T_X)

                # calculate probability of network following sequence

                p_seqs.append(ntwk.sequence_probability(test_sequence, r_0, xc_0, drives))

            p_seqs_all.append(p_seqs)

        p_seqs_alls.append(p_seqs_all)


    # MAKE PLOTS

    fig_size = (AX_SIZE[0], n_seqs * AX_SIZE[1])

    fig, axs = plt.subplots(
        n_seqs, 1, figsize=fig_size, facecolor='white',
        sharex=True, sharey=True, tight_layout=True)

    if n_seqs == 1:
        axs = [axs]

    labels = []

    for ax_ctr, (ax, p_seqs_all) in enumerate(zip(axs, p_seqs_alls)):

        for fixed_params, p_seqs in zip(fixed_params_all, p_seqs_all):

            ax.plot(G_BARS, p_seqs, lw=2)

            if ax_ctr == 0:

                labels.append('gw* = {}, gx* = {}, gd* = {}'.format(*fixed_params))

    for ax_ctr, ax in enumerate(axs):

        ax.set_title('sequence {} (length {})'.format(ax_ctr + 1, run_times[ax_ctr]))
        if ax_ctr == 0:

            axs[0].legend(labels, loc='best')

            axs[0].set_ylim(0, 1)

        if ax_ctr == n_seqs - 1:

            ax.set_xlabel('gbar')

        ax.set_ylabel('replay \n probability')

        set_fontsize(ax, FONT_SIZE)


def nonreplayable_sequence_example(
        SEED, GRID_SHAPE, LATERAL_SPREAD,
        G_BAR_EXAMPLE, G_BARS, G_W_STAR, G_X_STAR, G_D_STAR, T_X,
        DRIVEN_NODES, DRIVE_AMPLITUDE,
        FIG_SIZE, FONT_SIZE):

    # RUN SIMULATION

    np.random.seed(SEED)

    # make weight matrix

    w = make_feed_forward_grid_weights(GRID_SHAPE, LATERAL_SPREAD)
    n_nodes = w.shape[0]

    g_w = G_BAR_EXAMPLE * G_W_STAR
    g_x = G_BAR_EXAMPLE * G_X_STAR
    g_d = G_BAR_EXAMPLE * G_D_STAR

    run_time_example = 2 * len(DRIVEN_NODES)
    run_time_prob = len(DRIVEN_NODES)

    # make network

    ntwk_example = network(w=w, g_w=g_w, g_x=g_x, g_d=g_d, t_x=T_X)

    # make drive sequence

    drives_example = np.zeros((run_time_example, n_nodes), dtype=float)

    for t, node in enumerate(DRIVEN_NODES):
        drives_example[t, node] = DRIVE_AMPLITUDE

    drives_example[len(DRIVEN_NODES), DRIVEN_NODES[0]] = DRIVE_AMPLITUDE

    r_0_example = np.zeros((n_nodes,))
    xc_0_example = np.zeros((n_nodes,))
    rs_example = ntwk_example.run(r_0=r_0_example, drives=drives_example, xc_0=xc_0_example)


    # DO PROBABILITY CALCULATION

    p_seqs = []

    for g_bar in G_BARS:

        g_w = g_bar * G_W_STAR
        g_x = g_bar * G_X_STAR
        g_d = g_bar * G_D_STAR

        ntwk = network(w=w, g_w=g_w, g_x=g_x, g_d=g_d, t_x=T_X)

        # set up initial state, initial hyperexcitability counter, and drives

        r_0_prob = np.zeros((n_nodes,))

        xc_0_prob = np.zeros((n_nodes,))
        xc_0_prob[DRIVEN_NODES] = np.arange(T_X - run_time_prob, T_X) + 1

        drives_prob = np.zeros((run_time_prob, n_nodes))
        drives_prob[0, DRIVEN_NODES[0]] = DRIVE_AMPLITUDE

        # calculate probability of network following sequence

        p_seqs.append(ntwk.sequence_probability(DRIVEN_NODES, r_0_prob, xc_0_prob, drives_prob))


    # MAKE PLOTS

    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for ax_ctr, ax in enumerate(axs):

        if ax_ctr == 0:

            # example trace

            fancy_raster(ax, rs_example, drives_example)

            ax.set_ylim(0, n_nodes)

            ax.set_xlabel('time step')
            ax.set_ylabel('ensemble')

        if ax_ctr == 1:

            # probability vs g_bar

            ax.plot(G_BARS, p_seqs, lw=2)

            ax.set_ylim(0, 1)

            ax.set_xlabel('gbar')
            ax.set_ylabel('sequence replay \n probability')

        set_fontsize(ax, FONT_SIZE)