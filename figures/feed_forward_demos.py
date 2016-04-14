from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from connectivity import feed_forward_grid
from network import SoftmaxWTAWithLingeringHyperexcitability as network
from plot import fancy_raster, set_fontsize


def replay_example(
        SEED, GRID_SHAPE, LATERAL_SPREAD, G_W, G_X, G_D, T_X,
        DRIVEN_NODES, DRIVE_AMPLITUDE,
        AX_SIZE, FONT_SIZE):

    # RUN SIMULATION

    np.random.seed(SEED)
    n_trials = len(DRIVEN_NODES)

    # make network

    w = feed_forward_grid(shape=GRID_SHAPE, spread=1) + feed_forward_grid(shape=GRID_SHAPE, spread=LATERAL_SPREAD)
    ntwk = network(w=w, g_w=G_W, g_x=G_X, g_d=G_D, t_x=T_X)

    # make drive sequence

    all_drives = []
    all_rs = []

    for driven_nodes_one_trial in DRIVEN_NODES:

        run_time = 2 * len(driven_nodes_one_trial)
        n_nodes = w.shape[0]

        drives = np.zeros((run_time, n_nodes), dtype=float)
        driven_nodes_flat = np.ravel_multi_index(np.transpose(driven_nodes_one_trial), GRID_SHAPE)

        for t, node in enumerate(driven_nodes_flat):
            drives[t, node] = DRIVE_AMPLITUDE

        drives[len(driven_nodes_one_trial), driven_nodes_flat[0]] = DRIVE_AMPLITUDE

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
        ax.set_ylabel('Ensemble')

        if ax_ctr == n_trials - 1:
            ax.set_xlabel('Time step')

        set_fontsize(ax, FONT_SIZE)