from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

from plot import set_fontsize


def er_capacity(
        PS, LS,
        FIG_SIZE, COLORS, FONT_SIZE):
    """
    Make the following plots to demonstrate the capacity of an ER network for reliably replayable
    sequences:

    Normalized probability that a randomly chosen sequence is reliably replayable.

    :param PS: array of probabilities from 0 to 1
    :param LS: sequence lengths to plot
    :param FIG_SIZE:
    :param FONT_SIZE:
    :return:
    """

    # for each L, calculate probability of reliably replayable path

    p_rrs_normed = []

    for l in LS:

        p_rr = (PS ** l) * ((1 - PS) ** ((l - 1) * (l - 2)))
        p_rr_normed = p_rr / p_rr.max()

        p_rrs_normed.append(p_rr_normed)

    # make plots

    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)

    for l, p_rr_normed, color in zip(LS, p_rrs_normed, COLORS):

        axs[0].plot(PS, p_rr_normed, lw=2, color=color)


    axs[0].set_xlim(0, 1)

    axs[0].set_xlabel('p')
    axs[0].set_ylabel('p(random path is RR)\n(normalized to max)')
    axs[0].set_title('Erdos-Renyi')

    axs[0].legend(LS)

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)