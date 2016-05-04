from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

import connectivity
import metrics
import network
from plot import set_fontsize


def single_time_point_decoding_vs_binary_weight_matrix(
        SEED,
        N_NODES, P_CONNECT, G_W, G_DS, G_D_EXAMPLE, N_TIME_POINTS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities. (when weight matrix is binary)
    """

    keys = ['matched', 'zero', 'half_matched', 'random', 'full']

    np.random.seed(SEED)

    # build original weight matrix and convert to drive transition probability distribution

    ws = {}

    ws['matched'] = connectivity.er_directed(N_NODES, P_CONNECT)
    p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(ws['matched'], G_W)

    # build the other weight matrices

    ws['zero'] = np.zeros((N_NODES, N_NODES), dtype=float)

    ws['half_matched'] = ws['matched'].copy()
    ws['half_matched'][np.random.rand(*ws['half_matched'].shape) < 0.5] = 0

    ws['random'] = connectivity.er_directed(N_NODES, P_CONNECT)

    ws['full'] = connectivity.er_directed(N_NODES, 1)

    # make networks

    ntwks = {}

    for key, w in ws.items():

        ntwks[key] = network.SoftmaxWTAWithLingeringHyperexcitability(
            w, g_w=G_W, g_x=0, g_d=None, t_x=0)

    # a few checks on their creation

    assert np.sum(ntwks['matched'].w != ntwks['zero'].w) == np.sum(ntwks['matched'].w)
    assert 0 < np.sum(ntwks['half_matched'].w) < 0.75 * np.sum(ntwks['matched'].w)

    # create sample drive sequence

    drives = np.zeros((N_TIME_POINTS, N_NODES))

    drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
    drives[0, drive_first] = 1

    for ctr in range(N_TIME_POINTS - 1):

        drive_last = np.argmax(drives[ctr])
        drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

        drives[ctr + 1, drive_next] = 1

    drive_seq = np.argmax(drives, axis=1)

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {key: [] for key in keys}
    decoding_results_examples = {}

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for g_d in list(G_DS) + [G_D_EXAMPLE]:

        # set drive gain in all networks and run them

        for key, ntwk in ntwks.items():

            ntwk.g_d = g_d

            rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

            decoding_results = (rs_seq == drive_seq)
            decoding_accuracies[key].append(np.mean(decoding_results))

            if g_d == G_D_EXAMPLE:
                decoding_results_examples[key] = decoding_results

    # plot things

    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for key, color in zip(keys, COLORS):

        axs[0].plot(G_DS, decoding_accuracies[key][:-1], c=color, lw=2)

    axs[0].set_xlim(G_DS[0], G_DS[-1])
    axs[0].set_ylim(0, 1.1)

    axs[0].set_xlabel('g_d')
    axs[0].set_ylabel('decoding accuracy')

    axs[0].set_title('single time-point decoding accuracy for different network connectivities')

    axs[0].legend(keys, loc='best')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        decoding_results = decoding_results_examples[key]
        axs[1].plot(decoding_results + 0.01 * ctr, c=color, lw=2)

    axs[1].set_xlim(0, 140)
    axs[1].set_ylim(0, 1.1)

    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('correct decoding')

    axs[1].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)


def single_time_point_decoding_vs_nary_weight_matrix(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS, G_W, G_DS, G_D_EXAMPLE, N_TIME_POINTS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities. (when weight matrix is nonbinary)
    """

    keys = ['matched', 'zero', 'half_matched', 'random', 'full']

    np.random.seed(SEED)

    # build original weight matrix and convert to drive transition probability distribution

    ws = {}

    ws['matched'] = connectivity.er_directed_nary(N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS)
    p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(ws['matched'], G_W)

    # build the other weight matrices

    ws['zero'] = np.zeros((N_NODES, N_NODES), dtype=float)

    ws['half_matched'] = ws['matched'].copy()
    ws['half_matched'][np.random.rand(*ws['half_matched'].shape) < 0.5] = 0

    ws['random'] = connectivity.er_directed_nary(N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS)

    ws['full'] = connectivity.er_directed_nary(N_NODES, 1, STRENGTHS, P_STRENGTHS)

    # make networks

    ntwks = {}

    for key, w in ws.items():

        ntwks[key] = network.SoftmaxWTAWithLingeringHyperexcitability(
            w, g_w=G_W, g_x=0, g_d=None, t_x=0)

    # perform a few checks

    assert np.sum(np.abs(ws['zero'])) == 0

    for strength in STRENGTHS:

        assert np.sum(ws['matched'] == strength) > 0

    # create sample drive sequence

    drives = np.zeros((N_TIME_POINTS, N_NODES))

    drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
    drives[0, drive_first] = 1

    for ctr in range(N_TIME_POINTS - 1):

        drive_last = np.argmax(drives[ctr])
        drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

        drives[ctr + 1, drive_next] = 1

    drive_seq = np.argmax(drives, axis=1)

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {key: [] for key in keys}
    decoding_results_examples = {}

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for g_d in list(G_DS) + [G_D_EXAMPLE]:

        # set drive gain in all networks and run them

        for key, ntwk in ntwks.items():

            ntwk.g_d = g_d

            rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

            decoding_results = (rs_seq == drive_seq)
            decoding_accuracies[key].append(np.mean(decoding_results))

            if g_d == G_D_EXAMPLE:
                decoding_results_examples[key] = decoding_results

    # plot things

    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for key, color in zip(keys, COLORS):

        axs[0].plot(G_DS, decoding_accuracies[key][:-1], c=color, lw=2)

    axs[0].set_xlim(G_DS[0], G_DS[-1])
    axs[0].set_ylim(0, 1.1)

    axs[0].set_xlabel('g_d')
    axs[0].set_ylabel('decoding accuracy')

    axs[0].set_title('single time-point decoding accuracy for different network connectivities')

    axs[0].legend(keys, loc='best')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        decoding_results = decoding_results_examples[key]
        axs[1].plot(decoding_results + 0.01 * ctr, c=color, lw=2)

    axs[1].set_xlim(0, 140)
    axs[1].set_ylim(0, 1.1)

    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('correct decoding')

    axs[1].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)