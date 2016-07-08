from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

from network import LIFExponentialSynapsesModel
from plot import multivariate_same_axis
from plot import set_fontsize


def sequence_replay_branching_chain_demo(
        SEED, V_REST, V_TH, V_RESET, REFRAC_PER, TAU_M,
        TAUS_SYN, V_REVS_SYN,
        CHAIN_LENGTH_PRE, CHAIN_LENGTH_POST,
        W_PP, W_PI, W_PM, W_IP, W_II, W_MP, W_MM,
        BKGD_STARTS, BKGD_ENDS, BKGD_FRQS, BKGD_STRENS,
        DRIVE_ORDERS, DRIVE_STARTS, DRIVE_STRENS, DRIVE_DURS, DRIVE_FRQS, DRIVE_ITVS,
        REPLAY_TRIGGER_TIMES, REPLAY_TRIGGER_STRENS,
        MEMORY_RESET_STARTS, MEMORY_RESET_ENDS, MEMORY_RESET_STRENS, MEMORY_RESET_FRQS,
        SIM_DURATION, DT,
        FIG_SIZE, FONT_SIZE):

    syns = TAUS_SYN.keys()

    # build drive arrays

    np.random.seed(SEED)

    n_steps = int(SIM_DURATION / DT)
    n_primary_cells = CHAIN_LENGTH_PRE + 2 * CHAIN_LENGTH_POST
    n_cells = 2 * n_primary_cells + 1

    drives = {syn: np.zeros((n_steps, n_cells)) for syn in syns}

    for syn in syns:

        for drive_ctr in range(len(DRIVE_ORDERS)):

            # drive one path through network with a sequence of volleys

            start = DRIVE_STARTS[drive_ctr][syn]
            dur = DRIVE_DURS[drive_ctr][syn]
            frq = DRIVE_FRQS[drive_ctr][syn]
            itv = DRIVE_ITVS[drive_ctr][syn]

            if frq > 0:

                period = 1. / frq

            else:

                period = np.inf

            for cell_ctr, cell in enumerate(DRIVE_ORDERS[drive_ctr]):

                volley_start = start + cell_ctr * itv
                volley_end = volley_start + dur
                volley_times = np.arange(volley_start, volley_end, period)

                volley_times_idx = np.round(volley_times / DT).astype(int)

                drives[syn][volley_times_idx, cell] += DRIVE_STRENS[drive_ctr][syn]

            # drive first neuron with replay trigger

            replay_trigger_time = REPLAY_TRIGGER_TIMES[drive_ctr][syn]
            replay_trigger_time_idx = np.round(replay_trigger_time / DT).astype(int)

            drives[syn][replay_trigger_time_idx, DRIVE_ORDERS[drive_ctr][0]] += \
                REPLAY_TRIGGER_STRENS[drive_ctr][syn]

        for mr_ctr in range(len(MEMORY_RESET_STARTS)):

            # reset memory units

            mr_start = MEMORY_RESET_STARTS[mr_ctr][syn]
            mr_end = MEMORY_RESET_ENDS[mr_ctr][syn]
            mr_stren = MEMORY_RESET_STRENS[mr_ctr][syn]
            mr_frq = MEMORY_RESET_FRQS[mr_ctr][syn]

            if mr_frq > 0:

                mr_period = 1. / mr_frq

            else:

                mr_period = np.inf

            mr_times = np.arange(mr_start, mr_end, mr_period)
            mr_times_idx = np.round(mr_times / DT).astype(int)

            drives[syn][mr_times_idx, n_primary_cells:2*n_primary_cells] += mr_stren

        # add background

        for bkgd_ctr in range(len(BKGD_STARTS)):

            bkgd_start_idx = int(BKGD_STARTS[bkgd_ctr][syn] / DT)
            bkgd_end_idx = int(BKGD_ENDS[bkgd_ctr][syn] / DT)
            bkgd_frq = BKGD_FRQS[bkgd_ctr][syn]
            bkgd_stren = BKGD_STRENS[bkgd_ctr][syn]

            bkgd = bkgd_stren * np.random.poisson(
                bkgd_frq * DT, (bkgd_end_idx - bkgd_start_idx, n_primary_cells))

            drives[syn][bkgd_start_idx:bkgd_end_idx, :n_primary_cells] += bkgd

    # build weight matrices

    ws = {syn: np.zeros((n_cells, n_cells)) for syn in syns}

    for syn in syns:

        # branching chain

        ws[syn][range(1, n_primary_cells), range(0, n_primary_cells - 1)] = W_PP[syn]

        temp_idx = CHAIN_LENGTH_PRE + CHAIN_LENGTH_POST

        ws[syn][temp_idx, temp_idx - 1] = 0
        ws[syn][temp_idx, CHAIN_LENGTH_PRE - 1] = W_PP[syn]

        # to primary from inhibitory

        ws[syn][:n_primary_cells, -1] = W_PI[syn]

        # to primary from memory

        ws[syn][range(n_primary_cells), range(n_primary_cells, 2*n_primary_cells)] = W_PM[syn]

        # to inhibitory from primary

        ws[syn][-1, :n_primary_cells] = W_IP[syn]

        # to inhibitory from inhibitory

        ws[syn][-1, -1] = W_II[syn]

        # to memory from primary

        ws[syn][range(n_primary_cells, 2*n_primary_cells), range(n_primary_cells)] = W_MP[syn]

        # memory to memory

        ws[syn][range(n_primary_cells, 2*n_primary_cells), range(n_primary_cells, 2*n_primary_cells)] = W_MM[syn]

    # make network

    ntwk = LIFExponentialSynapsesModel(
        v_rest=V_REST, tau_m=TAU_M, taus_syn=TAUS_SYN, v_revs_syn=V_REVS_SYN,
        v_th=V_TH, v_reset=V_RESET, refrac_per=REFRAC_PER, ws=ws)

    # set initial conditions for variables

    initial_conditions = {
        'voltages': V_REST * np.ones((n_cells,)),
        'conductances': {syn: np.zeros((n_cells,)) for syn in syns},
        'refrac_ctrs': np.zeros((n_cells,)),
    }

    # set desired measurables

    record = ('voltages', 'spikes', 'conductances')

    # run simulation

    measurements = ntwk.run(initial_conditions, drives, DT, record=record)


    # MAKE PLOTS

    fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)

    # primary voltages

    primary_voltages = measurements['voltages'][:, :n_primary_cells]

    scales = [V_TH - V_REST]
    spacing = 2
    colors = ['k']

    multivariate_same_axis(
        axs[0], measurements['time'], [primary_voltages], scales, spacing, colors)

    axs[0].set_ylabel('neuron')
    axs[0].set_title('primary neuron voltages')

    memory_voltages = measurements['voltages'][:, n_primary_cells:2 * n_primary_cells]

    scales = [V_TH - V_REST]
    spacing = 2
    colors = ['b']

    multivariate_same_axis(
        axs[1], measurements['time'], [memory_voltages], scales, spacing, colors)

    axs[1].set_ylabel('neuron')
    axs[1].set_title('memory neuron voltages')

    # primary spikes

    primary_spikes = measurements['spikes'][:, :n_primary_cells].nonzero()

    axs[2].scatter(primary_spikes[0] * DT, primary_spikes[1], s=200, marker='|', c='k', lw=1)

    axs[2].set_ylim(-1, n_primary_cells)
    axs[2].set_ylabel('neuron')
    axs[2].set_title('primary neuron spikes')

    # memory spikes

    memory_spikes = measurements['spikes'][:, n_primary_cells:2*n_primary_cells].nonzero()

    axs[3].scatter(memory_spikes[0] * DT, memory_spikes[1], s=150, marker='|', c='b', lw=1)

    axs[3].set_xlim(0, SIM_DURATION)
    axs[3].set_ylim(-1, n_primary_cells)

    axs[3].set_xlabel('time (s)')
    axs[3].set_ylabel('neuron')
    axs[3].set_title('memory neuron spikes')

    for ax in axs:

        set_fontsize(ax, FONT_SIZE)

    return fig