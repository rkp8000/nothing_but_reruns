from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import connectivity
import metrics
import network
from plot import set_fontsize


def single_time_point_decoding_vs_binary_weight_matrix(
        SEED,
        N_NODES, P_CONNECT, G_W, G_DS, G_D_EXAMPLE, N_TIME_POINTS,
        N_TIME_POINTS_EXAMPLE, DECODING_SEQUENCE_LENGTHS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities. (when weight matrix is binary)
    """

    STRENGTHS = [1.]
    P_STRENGTHS = [1.]

    single_time_point_decoding_vs_nary_weight_matrix(
        SEED=SEED, N_NODES=N_NODES,
        P_CONNECT=P_CONNECT, STRENGTHS=STRENGTHS, P_STRENGTHS=P_STRENGTHS,
        G_W=G_W, G_DS=G_DS, G_D_EXAMPLE=G_D_EXAMPLE,
        N_TIME_POINTS=N_TIME_POINTS, N_TIME_POINTS_EXAMPLE=N_TIME_POINTS_EXAMPLE,
        DECODING_SEQUENCE_LENGTHS=DECODING_SEQUENCE_LENGTHS,
        FIG_SIZE=FIG_SIZE, COLORS=COLORS, FONT_SIZE=FONT_SIZE)


def single_time_point_decoding_vs_nary_weight_matrix(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS, G_W, G_DS, G_D_EXAMPLE,
        N_TIME_POINTS, N_TIME_POINTS_EXAMPLE, DECODING_SEQUENCE_LENGTHS,
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
    drive_seqs = {
        seq_len: metrics.gather_sequences(drive_seq, seq_len)
        for seq_len in DECODING_SEQUENCE_LENGTHS
    }

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {
        key: {
            seq_len: []
            for seq_len in DECODING_SEQUENCE_LENGTHS
        }
        for key in keys
    }

    decoding_results_examples = {}

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for g_d in list(G_DS) + [G_D_EXAMPLE]:

        # set drive gain in all networks and run them

        for key, ntwk in ntwks.items():

            ntwk.g_d = g_d

            rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

            # calculate decoding accuracy for this network for all specified sequence lengths

            for seq_len in DECODING_SEQUENCE_LENGTHS:

                rs_seq_staggered = metrics.gather_sequences(rs_seq, seq_len)

                decoding_results = np.all(rs_seq_staggered == drive_seqs[seq_len], axis=1)

                decoding_accuracies[key][seq_len].append(np.mean(decoding_results))

            if g_d == G_D_EXAMPLE:
                decoding_results_examples[key] = (rs_seq == drive_seq)

    # plot things

    n_seq_lens = len(DECODING_SEQUENCE_LENGTHS)

    fig = plt.figure(figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    # the top row will be decoding accuracies for all sequence lengths

    axs = [fig.add_subplot(2, n_seq_lens, ctr + 1) for ctr in range(n_seq_lens)]

    # the bottom row is example decoding accuracy time courses

    axs.append(fig.add_subplot(2, 1, 2))

    for ax, seq_len in zip(axs[:-1], DECODING_SEQUENCE_LENGTHS):

        for key, color in zip(keys, COLORS):

            ax.plot(G_DS, decoding_accuracies[key][seq_len][:-1], c=color, lw=2)

        ax.set_xlim(G_DS[0], G_DS[-1])
        ax.set_ylim(0, 1.1)

        ax.set_xlabel('g_d')

    axs[0].set_ylabel('decoding accuracy')

    for ax, seq_len in zip(axs[:-1], DECODING_SEQUENCE_LENGTHS):

        ax.set_title('Length {} sequences'.format(seq_len))

    axs[0].legend(keys, loc='best')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        decoding_results = decoding_results_examples[key]
        y_vals = 2 * ctr + decoding_results

        axs[-1].plot(y_vals, c=color, lw=2)
        axs[-1].axhline(2 * ctr, color='gray', lw=1, ls='--')

    axs[-1].set_xlim(0, N_TIME_POINTS_EXAMPLE)
    axs[-1].set_ylim(-1, 2 * len(keys) + 1)

    axs[-1].set_xlabel('time step')
    axs[-1].set_ylabel('correct decoding')

    axs[-1].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)


def single_time_point_decoding_vs_nary_weights_fixed_g_d(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS, G_W, G_D,
        N_TIME_POINTS, N_TRIALS, N_TIME_POINTS_EXAMPLE,
        DECODING_SEQUENCE_LENGTHS,
        FIG_SIZE, MARKER_SIZE, VISUAL_SCATTER, COLORS, FONT_SIZE):
    """
    Run several trials of networks driven by different stimuli, with different weight
    matrices relative to the stimulus transition matrix.
    """

    keys = ['matched', 'zero', 'random', 'half_matched']

    np.random.seed(SEED)

    # build original weight matrix and convert to drive transition probability distribution

    ws = {}

    ws['matched'] = connectivity.er_directed_nary(N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS)
    p_tr_drive, p_0_drive = metrics.softmax_prob_from_weights(ws['matched'], G_W)

    # build the other weight matrices

    ws['zero'] = np.zeros((N_NODES, N_NODES), dtype=float)

    ws['random'] = connectivity.er_directed_nary(N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS)

    rand_mask = np.random.rand(*ws['random'].shape) < 0.5
    ws['half_matched'] = ws['matched'].copy()
    ws['half_matched'][rand_mask] = ws['random'][rand_mask]

    # make networks

    ntwks = {}

    for key, w in ws.items():

        ntwks[key] = network.SoftmaxWTAWithLingeringHyperexcitability(
            w, g_w=G_W, g_x=0, g_d=G_D, t_x=0)

    # perform a few checks

    assert np.sum(np.abs(ws['zero'])) == 0

    for strength in STRENGTHS:

        assert np.sum(ws['matched'] == strength) > 0

    # create random drive sequences

    drivess = []
    drive_seqss = []

    for _ in range(N_TRIALS):

        drives = np.zeros((N_TIME_POINTS, N_NODES))

        drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
        drives[0, drive_first] = 1

        for ctr in range(N_TIME_POINTS - 1):

            drive_last = np.argmax(drives[ctr])
            drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

            drives[ctr + 1, drive_next] = 1

        drive_seq = np.argmax(drives, axis=1)
        drive_seqs = {
            seq_len: metrics.gather_sequences(drive_seq, seq_len)
            for seq_len in DECODING_SEQUENCE_LENGTHS
        }

        drivess.append(drives)
        drive_seqss.append(drive_seqs)

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {
        key: {
            seq_len: []
            for seq_len in DECODING_SEQUENCE_LENGTHS
        }
        for key in keys
    }

    decoding_results_example = {}

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    # run networks

    for key, ntwk in ntwks.items():

        # loop over trials

        for tr_ctr, (drives, drive_seqs) in enumerate(zip(drivess, drive_seqss)):

            rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

            # calculate decoding accuracy for this network for all specified sequence lengths

            for seq_len in DECODING_SEQUENCE_LENGTHS:

                rs_seq_staggered = metrics.gather_sequences(rs_seq, seq_len)

                decoding_results = np.all(rs_seq_staggered == drive_seqs[seq_len], axis=1)

                decoding_accuracies[key][seq_len].append(np.mean(decoding_results))

                if seq_len == 1 and tr_ctr == 0:

                    decoding_results_example[key] = decoding_results.flatten()


    # plot things

    n_seq_lens = len(DECODING_SEQUENCE_LENGTHS)

    # one subplot per sequence length, plus one for example decoding time-series

    fig = plt.figure(figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    axs = [fig.add_subplot(2, n_seq_lens, ctr) for ctr in range(1, n_seq_lens + 1)]
    axs.append(fig.add_subplot(2, 1, 2))

    for ax, seq_len in zip(axs[:-1], DECODING_SEQUENCE_LENGTHS):

        for ctr, (key, color) in enumerate(zip(keys, COLORS)):

            acc = decoding_accuracies[key][seq_len]

            y_vals = ctr * np.ones((len(acc),)) + np.random.normal(0, VISUAL_SCATTER, len(acc))

            ax.scatter(acc, y_vals, c=color, s=MARKER_SIZE, lw=0)

        ax.set_xlim(-.1, 1.1)
        ax.set_yticks(range(len(keys)))
        ax.set_xlabel('decoding accuracy')
        ax.set_title('length {} sequences'.format(seq_len))

        if ax is not axs[0]:

            ax.set_yticklabels(len(keys) * [''])

    axs[0].set_yticklabels(keys)

    axs[-2].legend(keys, loc='best')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        decoding_results = decoding_results_example[key]
        y_vals = 2 * ctr + decoding_results

        axs[-1].plot(y_vals, c=color, lw=2)
        axs[-1].axhline(2 * ctr, color='gray', lw=1, ls='--')

    axs[-1].set_xlim(0, N_TIME_POINTS_EXAMPLE)
    axs[-1].set_ylim(-1, 2 * len(keys) + 1)

    axs[-1].set_xlabel('time step')
    axs[-1].set_ylabel('correct decoding')

    axs[-1].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)


def spontaneous_vs_driven_dkl(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS, G_W, G_DS, N_TIME_POINTS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Compare the DKL of states vs and state transitions between spontaneous and driven activities
    for different weight matrices.
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

    # calculate spontaneous state and transition probabilities matrices

    states_spontaneous = {}
    transitions_spontaneous = {}

    for key, ntwk in ntwks.items():

        transitions_spontaneous[key], states_spontaneous[key] = \
            metrics.softmax_prob_from_weights(ntwk.w, G_W)

    # create sample drive sequence

    drives = np.zeros((N_TIME_POINTS, N_NODES))

    drive_first = np.random.choice(np.arange(N_NODES), p=p_0_drive.flatten())
    drives[0, drive_first] = 1

    for ctr in range(N_TIME_POINTS - 1):

        drive_last = np.argmax(drives[ctr])
        drive_next = np.random.choice(range(N_NODES), p=p_tr_drive[:, drive_last])

        drives[ctr + 1, drive_next] = 1

    # run simulations and calculate DKLs

    state_dkls = {key: [] for key in keys}
    transition_dkls = {key: [] for key in keys}

    r_0 = np.zeros((N_NODES,))
    xc_0 = np.zeros((N_NODES,))

    for g_d in G_DS:

        for key, ntwk in ntwks.items():

            ntwk.g_d = g_d

            rs_seq = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives).argmax(axis=1)

            # calculate state probabilities from sequence
            states_driven = metrics.occurrence_count(rs_seq, states=np.arange(N_NODES))
            states_driven /= states_driven.sum()

            state_dkls[key].append(
                stats.entropy(states_driven, states_spontaneous[key]))

            # calculate transition probabilities from sequence

            transitions_driven = metrics.transition_count(rs_seq, states=np.arange(N_NODES))
            transitions_driven /= transitions_driven.sum()

            transition_dkls[key].append(
                metrics.transition_dkl(transitions_driven, transitions_spontaneous[key]))


    # make plots

    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for key, color in zip(keys, COLORS):

        axs[0].plot(G_DS, state_dkls[key], color=color, lw=2)
        axs[1].plot(G_DS, transition_dkls[key], color=color, lw=2)

    for ctr, ax in enumerate(axs):

        if ctr == 0:

            ax.legend(keys, loc='best')

        ax.set_xlabel('g_d')

        if ctr == 0:

            ax.set_ylabel('spontaneous vs. driven DKL')

            ax.set_title('state probabilities')

        else:

            ax.set_title('transition probabilities')

        set_fontsize(ax, FONT_SIZE)


def single_time_point_decoding_with_random_occlusion(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS,
        G_W, P_OCCLUSION, OCCLUSION_FACTOR, G_DS, G_D_EXAMPLE, N_TIME_POINTS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities. (when weight matrix is nonbinary).
    This is the case in which the stimulus is sometimes randomly occluded.
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

    # randomly occlude some of the stimuli

    occlusion_mask = np.random.rand(len(drives)) < P_OCCLUSION

    if P_OCCLUSION > 0:

        assert len(occlusion_mask) == len(drives)
        assert 0 < occlusion_mask.sum() < len(occlusion_mask)

    drives[occlusion_mask, :] *= (1 - OCCLUSION_FACTOR)

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {key: [] for key in keys}
    decoding_accuracies_occluded = {key: [] for key in keys}
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

            decoding_results_occluded = (rs_seq[occlusion_mask] == drive_seq[occlusion_mask])
            decoding_accuracies_occluded[key].append(np.mean(decoding_results_occluded))

            if g_d == G_D_EXAMPLE:
                decoding_results_examples[key] = decoding_results

    # plot things

    fig, axs = plt.subplots(3, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for key, color in zip(keys, COLORS):

        axs[0].plot(G_DS, decoding_accuracies[key][:-1], c=color, lw=2)

    axs[0].set_xlim(G_DS[0], G_DS[-1])
    axs[0].set_ylim(0, 1.1)

    axs[0].set_xlabel('g_d')
    axs[0].set_ylabel('decoding accuracy')

    axs[0].set_title('single time-point decoding accuracy for different network connectivities')

    axs[0].legend(keys, loc='best')

    for key, color in zip(keys, COLORS):

        axs[1].plot(G_DS, decoding_accuracies_occluded[key][:-1], c=color, lw=2)

    axs[1].set_xlim(G_DS[0], G_DS[-1])
    axs[1].set_ylim(0, 1.1)

    axs[1].set_xlabel('g_d')
    axs[1].set_ylabel('decoding accuracy')

    axs[1].set_title('decoding accuracy for occluded points only')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        decoding_results = decoding_results_examples[key]
        axs[2].plot(decoding_results + 0.01 * ctr, c=color, lw=2)

    axs[2].set_xlim(0, 140)
    axs[2].set_ylim(0, 1.1)

    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('correct decoding')

    axs[2].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)


def single_time_point_decoding_with_random_spreading(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS,
        G_W, P_STIM_SPREAD, STIM_SPREAD_FACTORS, G_DS, G_D_EXAMPLE, N_TIME_POINTS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Explore how the ability to decode an external drive at a single time point depends on the alignment
    of the weight matrix with the stimulus transition probabilities. (when weight matrix is nonbinary).
    This is the case in which the stimulus is sometimes randomly occluded.
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

    # randomly mix up some of the stimuli

    spread_mask = np.random.rand(len(drives)) < P_STIM_SPREAD

    assert len(spread_mask) == len(drives)

    if P_STIM_SPREAD > 0:

        assert 0 < spread_mask.sum() < len(spread_mask)

    spread_size = len(STIM_SPREAD_FACTORS) - 1

    for ctr, (mask_value, drive_el) in enumerate(zip(spread_mask, drive_seq)):

        if mask_value:

            # get random set of elements to spread stimulus across

            spread_idxs = np.random.choice(
                range(0, drive_el) + range(drive_el + 1, N_NODES),
                size=spread_size,
                replace=False)

            # spread stimulus across these nodes

            drives[ctr, drive_el] = STIM_SPREAD_FACTORS[0]

            drives[ctr, spread_idxs] = STIM_SPREAD_FACTORS[1:]

    # loop through various external drive gains and calculate how accurate the stimulus decoding is

    decoding_accuracies = {key: [] for key in keys}
    decoding_accuracies_spread = {key: [] for key in keys}
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

            decoding_results_spread = (rs_seq[spread_mask] == drive_seq[spread_mask])
            decoding_accuracies_spread[key].append(np.mean(decoding_results_spread))

            if g_d == G_D_EXAMPLE:
                decoding_results_examples[key] = decoding_results

    # plot things

    fig, axs = plt.subplots(3, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for key, color in zip(keys, COLORS):

        axs[0].plot(G_DS, decoding_accuracies[key][:-1], c=color, lw=2)

    axs[0].set_xlim(G_DS[0], G_DS[-1])
    axs[0].set_ylim(0, 1.1)

    axs[0].set_xlabel('g_d')
    axs[0].set_ylabel('decoding accuracy')

    axs[0].set_title('single time-point decoding accuracy for different network connectivities')

    axs[0].legend(keys, loc='best')

    for key, color in zip(keys, COLORS):

        axs[1].plot(G_DS, decoding_accuracies_spread[key][:-1], c=color, lw=2)

    axs[1].set_xlim(G_DS[0], G_DS[-1])
    axs[1].set_ylim(0, 1.1)

    axs[1].set_xlabel('g_d')
    axs[1].set_ylabel('decoding accuracy')

    axs[1].set_title('decoding accuracy for stim-spread time points only')

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        decoding_results = decoding_results_examples[key]
        axs[2].plot(decoding_results + 0.01 * ctr, c=color, lw=2)

    axs[2].set_xlim(0, 140)
    axs[2].set_ylim(0, 1.1)

    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('correct decoding')

    axs[2].set_title('example decoder time course')

    for ax in axs:
        set_fontsize(ax, FONT_SIZE)