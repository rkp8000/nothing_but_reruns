from __future__ import division, print_function
import matplotlib.pyplot as plt
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


def multivariate_same_axis(ax, ts, data, scales, spacing, colors, z_orders=None, **plot_kwargs):
    """
    Plot multiple multivariate time-series on the same axis object.

    :param ax: axis object
    :param ts: 1D time array
    :param data: list of arrays; each array should have time steps along dim 0
    :param scales: scaling parameters
    :param spacing: spacing between variables
    :param colors: colors for each time-series
    :param z_orders: orders of appearance on page
    :return y-offsets for each row
    """

    if z_orders is None:

        z_orders = range(len(data))[::-1]

    # loop over data sets

    for datum, scale, color, z_order in zip(data, scales, colors, z_orders):

        # loop over columns of data set

        for ctr, col in enumerate(datum.T):

            y_coords = col / scale + ctr * spacing

            ax.plot(ts, y_coords, color=color, zorder=z_order, **plot_kwargs)

    return np.arange(len(datum.T)) * spacing


def firing_rate_heat_map(ax, dt, rs, vmin=0, vmax=1):
    """
    Plot a heat map of firing rates

    :param ax: axis object
    """

    ax.matshow(
        rs.T, origin='lower',
        interpolation='nearest', cmap=plt.cm.hot, vmin=vmin, vmax=vmax)

    ax.set_xticklabels(ax.get_xticks() * dt)
    ax.set_aspect('auto')
    ax.xaxis.tick_bottom()