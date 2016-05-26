"""
Figures demonstrating dynamical systems implementation of replay.
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import connectivity
import network
from plot import set_fontsize

plt.style.use('ggplot')


def _build_tree_structure_demo_connectivity(
        branch_length, w_pp, w_mp, w_pm, w_mm):

    cc = np.concatenate

    # calculate each branch

    idx_offset = 3 * branch_length

    branch_0_0 = 2 * np.arange(branch_length + 1)
    branch_0_1 = np.arange(2 * branch_length + 1, 3 * branch_length + 1)

    branch_0 = cc([branch_0_0, branch_0_1])

    branch_2_0 = 2 * np.arange(branch_length + 1)
    branch_2_1 = 2 * np.arange(branch_length)[::-1] + 1

    branch_2 = cc([branch_2_0, branch_2_1])

    branches = [branch_0, -branch_0, branch_2, -branch_2]
    branches = [branch + idx_offset for branch in branches]

    # set up network

    n_principal_nodes = 6 * branch_length + 1

    # make list of all principal-to-principal connections

    cxns_pp = []

    for branch in branches:

        cxns = np.transpose([branch[1:], branch[:-1]])
        cxns_pp.extend([tuple(cxn) for cxn in cxns])

    # make principal-to-principal connection mask

    principal_mask = np.zeros((n_principal_nodes, n_principal_nodes))

    for cxn in cxns_pp:
        principal_mask[cxn] = 1

    # build final weight matrix

    w = connectivity.basic_adlib(
        principal_mask, w_pp=w_pp, w_mp=w_mp, w_pm=w_pm, w_mm=w_mm)

    return w, branches


def tree_structure_replay_demo(
        SEED,
        TAU, V_REST, V_TH, GAIN, NOISE, DT,
        BRANCH_LENGTH, W_PP, W_MP, W_PM, W_MM,
        PULSE_START, PULSE_DURATION, INTER_PULSE_INTERVAL, PULSE_HEIGHT,
        BRANCH_ORDER, INTER_TRAIN_INTERVAL, REPLAY_WAIT_TIME,
        RESET_PULSE_START, RESET_PULSE_DURATION, RESET_PULSE_HEIGHT,
        FIG_SIZE, VERT_SPACING, FONT_SIZE):
    """
    Demo sequential replay in a network with a basic tree-structure.

    There are always 4 unique paths from the root of the tree to the ends
    of the branches.

    The network with branch length 3 has the following tree structure:

    * -< * -< * -< * -< * -< * -< *
     \              \
     \               -< * -< * -< *
     \
      -< * -< * -< * -< * -< * -< *
                    \
                     -< * -< * -< *

    :param SEED: RNG seed
    :param TAU: single node time constant
    :param V_REST: resting potential
    :param V_TH: threshold potential
    :param GAIN: gain
    :param NOISE: noise level
    :param DT: integration time interval
    :param BRANCH_LENGTH: how many nodes per branch
    :param W_PP: principal-to-principal weight
    :param W_MP: principal-to-memory weight
    :param W_PM: memory-to-principal weight
    :param W_MM: memory-to-memory weight
    :param PULSE_START: pulse start time
    :param PULSE_DURATION: pulse duration
    :param INTER_PULSE_INTERVAL: time between consecutive pulse starts
    :param PULSE_HEIGHT: pulse height
    :param BRANCH_ORDER: which order to stimulate branches in
    :param INTER_TRAIN_INTERVAL: interval between pulse train starts
    :param REPLAY_WAIT_TIME: time between last pulse start and trigger pulse
    :param RESET_PULSE_START: time of reset pulse, relative to train start
    :param RESET_PULSE_DURATION: duration of reset pulse
    :param RESET_PULSE_HEIGHT: height of reset pulse
    :param FIG_SIZE: figure size
    :param VERT_SPACING: spacing between neuron firing rate traces
    :param FONT_SIZE: font size
    """

    np.random.seed(SEED)

    w, branches = _build_tree_structure_demo_connectivity(
        branch_length=BRANCH_LENGTH,
        w_pp=W_PP, w_mp=W_MP, w_pm=W_PM, w_mm=W_MM)

    n_nodes = w.shape[0]

    # build final network

    ntwk = network.RateBasedModel(
        taus=TAU*np.ones((n_nodes,)),
        v_rests=V_REST*np.ones((n_nodes,)),
        v_ths=V_TH*np.ones((n_nodes,)),
        gains=GAIN*np.ones((n_nodes,)),
        noises=NOISE*np.ones((n_nodes,)),
        w=w)

    # build drive sequences

    for branch_idx in BRANCH_ORDER:

        if branch_idx in [0, 1]:

            pass



def chain_propagation_demo(
        N_PRINCIPAL_NODES, FIG_SIZE, FONT_SIZE):
    """
    Demo propagation of activity through a rate-based model.

    :param N_NODES: number of nodes
    :param FIG_SIZE: figure size
    :param FONT_SIZE: font size
    """

    # choose parameters that we know work for successful propagation

    SEED = 0

    TAUS = 0.03 * np.ones((2 * N_PRINCIPAL_NODES,))
    V_RESTS = -0.06 * np.ones((2 * N_PRINCIPAL_NODES,))
    V_THS = -0.02 * np.ones((2 * N_PRINCIPAL_NODES,))
    GAINS = 100 * np.ones((2 * N_PRINCIPAL_NODES,))
    NOISES = .060 * np.ones((2 * N_PRINCIPAL_NODES,))

    W_PP = 0.08
    W_MP = 0
    W_PM = 0
    W_MM = 0

    PULSE_START = 0.2
    PULSE_HEIGHT = 1.5
    PULSE_END = 0.25

    SIM_DURATION = 3
    DT = 0.001

    VERT_SPACING = 1

    np.random.seed(SEED)

    # make connectivity matrix

    principal_mask = np.zeros(
        (N_PRINCIPAL_NODES, N_PRINCIPAL_NODES), dtype=float)

    for node in range(N_PRINCIPAL_NODES - 1):

        principal_mask[node + 1, node] = 1

    w = connectivity.basic_adlib(
        principal_connectivity_mask=principal_mask,
        w_pp=W_PP, w_mp=W_MP, w_pm=W_PM, w_mm=W_MM)

    # make network

    ntwk = network.RateBasedModel(
        taus=TAUS, v_rests=V_RESTS, v_ths=V_THS, gains=GAINS, noises=NOISES,
        w=w)

    # make pulse input

    n_time_steps = int(SIM_DURATION // DT)

    drives = np.zeros((n_time_steps, 2 * N_PRINCIPAL_NODES), dtype=float)
    drives[int(PULSE_START // DT):int(PULSE_END // DT), 0] = PULSE_HEIGHT

    v_0s = V_RESTS.copy()

    vs, rs = ntwk.run(v_0s, drives, dt=DT)

    # make figure

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    # plot input

    ts = np.arange(n_time_steps) * DT

    ax.plot(ts, (drives[:, 0] > 0).astype(float) + VERT_SPACING, c='g', lw=2)

    for node in range(N_PRINCIPAL_NODES):

        # plot firing rates

        y_offset = -node * VERT_SPACING

        ax.plot(ts, rs[:, node] + y_offset, c='k', lw=2)
        ax.axhline(y_offset, color='gray', ls='--')

    ax.set_yticks(-np.arange(-1, N_PRINCIPAL_NODES) * VERT_SPACING)
    ax.set_yticklabels(['input'] + range(N_PRINCIPAL_NODES))

    ax.set_xlabel('time (s)')
    ax.set_ylabel('node')

    set_fontsize(ax, FONT_SIZE)
