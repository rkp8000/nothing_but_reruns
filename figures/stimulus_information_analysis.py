from __future__ import division, print_function
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import connectivity
import metrics
import network
from plot import set_fontsize


def mutual_info_past_stim_current_sequence_vs_weights(
        SEED,
        N_NODES, P_CONNECT, STRENGTHS, P_STRENGTHS, G_W, G_X, G_D, T_X,
        PAST_SEQUENCE_LENGTH, CURRENT_SEQUENCE_LENGTH,
        MC_SAMPLE_SIZES, N_TRIALS,
        FIG_SIZE, VISUAL_SCATTER, MARKER_SIZE, COLORS, FONT_SIZE):
    """
    Estimate the mutual information between a past stimulus sequence and a current activity sequence;
    do this for a few different weight matrices.
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
            w, g_w=G_W, g_x=G_X, g_d=G_D, t_x=T_X)

    # perform a few checks

    assert np.sum(np.abs(ws['zero'])) == 0

    for strength in STRENGTHS:

        assert np.sum(ws['matched'] == strength) > 0

    info_estimates = {key: [] for key in keys}

    for key, ntwk in ntwks.items():

        for _ in range(N_TRIALS):

            info_estimates[key].append(
                metrics.mutual_info_past_stim_current_activity(
                    ntwk=ntwk, p_tr_stim=p_tr_drive, p_0_stim=p_0_drive,
                    past_seq_length=PAST_SEQUENCE_LENGTH,
                    current_seq_length=CURRENT_SEQUENCE_LENGTH,
                    mc_samples=MC_SAMPLE_SIZES))

    # get t- and p-values across populations

    t_vals, p_vals = metrics.multi_pop_stats_matrix(
        stat_fun=stats.ttest_ind,
        pop_names=keys,
        pops=[info_estimates[key] for key in keys])

    # display tables of statistics and p-values

    print('pairwise t-values')
    display(t_vals)

    print('pairwise p-values')
    display(p_vals)


    # plot things

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, facecolor='white', tight_layout=True)

    for ctr, (key, color) in enumerate(zip(keys, COLORS)):

        infos = info_estimates[key]

        y_vals = ctr * np.ones((N_TRIALS,)) + np.random.normal(0, VISUAL_SCATTER, N_TRIALS)

        ax.scatter(infos, y_vals, c=color, s=MARKER_SIZE, lw=0)

    ax.set_yticks(range(len(keys)))

    ax.set_xlabel('information')
    ax.set_yticklabels(keys)

    ax.legend(keys, loc='upper left')

    set_fontsize(ax, FONT_SIZE)

