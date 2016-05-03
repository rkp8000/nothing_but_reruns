from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

import connectivity
import metrics
import network
from plot import set_fontsize


def single_time_point_decoding_vs_weight_matrix(
        SEED,
        N_NODES, P_CONNECT, G_W, G_DS, G_D_EXAMPLE, N_TIME_POINTS,
        FIG_SIZE, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities.

    """

    np.random.seed(SEED)

    # build weight matrix and convert to drive transition probability distribution

    w_matched = connectivity.er_directed(N_NODES, P_CONNECT)
    p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(w_matched, G_W)

    # create sample drive sequence

    drives = np.zeros((N_TIME_POINTS, N_NODES))

    drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
    drives[0, drive_first] = 1

    for ctr in range(N_TIME_POINTS - 1):

        drive_last = np.argmax(drives[ctr])
        drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

        drives[ctr + 1, drive_next] = 1

    drive_seq = np.argmax(drives, axis=1)

    # build network using this weight matrix

    ntwk_matched = network.SoftmaxWTAWithLingeringHyperexcitability(
        w_matched, g_w=G_W, g_x=0, g_d=None, t_x=0)

    # build control network with no connections

    w_control = np.zeros((N_NODES, N_NODES), dtype=float)

    ntwk_control = network.SoftmaxWTAWithLingeringHyperexcitability(
        w_control, g_w=G_W, g_x=0, g_d=None, t_x=0)

    assert np.sum(ntwk_matched.w != ntwk_control.w) == np.sum(ntwk_matched.w)

    # build negative control network with random connections independent of drive transitions

    w_random = connectivity.er_directed(N_NODES, P_CONNECT)

    ntwk_random = network.SoftmaxWTAWithLingeringHyperexcitability(
        w_random, g_w=G_W, g_x=0, g_d=None, t_x=0)

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracy_matched = []
    decoding_accuracy_control = []
    decoding_accuracy_random = []

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for g_d in G_DS:

        # set drives in matched and control network

        ntwk_matched.g_d = g_d
        ntwk_control.g_d = g_d
        ntwk_random.g_d = g_d

        # present drives to networks

        rs_seq_matched = ntwk_matched.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)
        rs_seq_control = ntwk_control.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)
        rs_seq_random = ntwk_random.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

        decoding_results_matched = (rs_seq_matched == drive_seq)
        decoding_results_control = (rs_seq_control == drive_seq)
        decoding_results_random = (rs_seq_random == drive_seq)

        decoding_accuracy_matched.append(np.mean(decoding_results_matched))
        decoding_accuracy_control.append(np.mean(decoding_results_control))
        decoding_accuracy_random.append(np.mean(decoding_results_random))

    # run example

    ntwk_matched.g_d = G_D_EXAMPLE
    ntwk_control.g_d = G_D_EXAMPLE
    ntwk_random.g_d = G_D_EXAMPLE

    rs_seq_matched = ntwk_matched.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)
    rs_seq_control = ntwk_control.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)
    rs_seq_random = ntwk_random.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

    decoding_results_matched = (rs_seq_matched == drive_seq)
    decoding_results_control = (rs_seq_control == drive_seq)
    decoding_results_random = (rs_seq_random == drive_seq)

    # plot things

    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    axs[0].plot(G_DS, decoding_accuracy_matched, c='k', lw=2)
    axs[0].plot(G_DS, decoding_accuracy_control, c='b', lw=2)
    axs[0].plot(G_DS, decoding_accuracy_random, c='r', lw=2)

    axs[0].set_xlim(G_DS[0], G_DS[-1])
    axs[0].set_ylim(0, 1.1)

    axs[0].set_xlabel('g_d')
    axs[0].set_ylabel('decoding accuracy')

    axs[0].set_title('g_w = {}'.format(G_W))

    axs[0].legend(['matched', 'zero', 'random'], loc='best')

    axs[1].plot(decoding_results_matched, c='k', lw=2)
    axs[1].plot(decoding_results_control + .01, c='b', lw=2)
    axs[1].plot(decoding_results_random + .02, c='r', lw=1)

    axs[1].set_xlim(0, 100)
    axs[1].set_ylim(0, 1.1)

    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('correct decoding')

    axs[1].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)