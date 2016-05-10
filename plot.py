from __future__ import division, print_function
import numpy as np


def set_fontsize(ax, fontsize):
    """Set fontsize of all axis text objects to specified value."""

    for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        txt.set_fontsize(fontsize)

    legend = ax.get_legend()
    if legend:
        for txt in legend.get_texts():
            txt.set_fontsize(fontsize)


def fancy_raster(ax, spikes, drives, spike_marker_size=20, drive_marker_base_size=100):
    """
    Plot raster plots where each spike is a small dark solid circle and each drive is a surrounding hollow brighter
    circle with a radius that increases with drive strength.
    :param ax: axis object
    :param spikes:
    :param drives:
    :param spike_marker_size: how big the spike circle is
    :param drive_marker_base_size: how big the drive circle is for a drive amplitude of 1
    """

    spike_times, spike_rows = spikes.nonzero()

    ax.scatter(spike_times, spike_rows, c='k', s=spike_marker_size, zorder=1)

    if drives is not None:
        drive_times, drive_rows = drives.nonzero()
        drive_strengths = drive_marker_base_size * np.array([drives[t, r] for t, r in zip(drive_times, drive_rows)])

        ax.scatter(
            drive_times, drive_rows, s=drive_strengths,
            lw=1.2, alpha=.7, facecolors='none', edgecolors='r', zorder=-1
        )


def fancy_raster_arrows_above(ax, spikes, drives, spike_marker_size=20, arrow_marker_size=50, rise=6):
    """
    Plot raster plot, where certain spikes can be marked with a star above them.
    :param ax:
    :param spikes:
    :param drives:
    :param rise: how high above spike markers stars should be
    :return:
    """

    spike_times, spike_rows = spikes.nonzero()

    ax.scatter(spike_times, spike_rows, c='k', s=spike_marker_size, zorder=1)

    if drives is not None:

        drive_times, drive_rows = drives.nonzero()

        drive_rows += rise

        ax.scatter(
            drive_times, drive_rows, s=arrow_marker_size, marker='v',
            facecolor='r', edgecolor='none')