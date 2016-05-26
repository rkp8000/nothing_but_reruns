"""
Figures demonstrating dynamical systems implementation of replay.
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

import connectivity
import network
from plot import set_fontsize

plt.style.use('ggplot')


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
