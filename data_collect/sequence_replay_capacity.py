from __future__ import division, print_function
import logging

import networkx as nx
import numpy as np
from sqlalchemy import create_engine

import binary_graphs
import metrics

from data_collect import create_database
from data_collect import create_session
from data_collect import start_database_modification_test
from data_collect import _models


def er_capacity(
        session,
        N_NODESS, P_CONNECTS, N_NETWORKSS, SEQ_LENGTHS):
    """
    Calculate the replay capacity for a set of directed ER (Erdos-Renyi) networks.
    """

    for p_connect, n_networks in zip(P_CONNECTS, N_NETWORKSS):

        message = 'Running {} networks per network size with p_connect = {}.'.format(n_networks, p_connect)
        logging.info(message)

        for n_nodes in N_NODESS:

            message = 'Running networks of size {}.\n'.format(n_nodes)
            logging.info(message)

            for trial_ctr in range(n_networks):

                message = 'Building network # {}.\n'.format(trial_ctr + 1)
                logging.info(message)

                g = binary_graphs.erdos_renyi(n_nodes, p_connect)

                n_seqss_replayable = []
                n_seqss_non_replayable = []

                for seq_length in SEQ_LENGTHS:

                    message = 'Collecting sequences of length {}.'.format(seq_length)
                    if seq_length == SEQ_LENGTHS[-1]:
                        message += '\n'
                    logging.info(message)

                    seqs_replayable, seqs_non_replayable = metrics.replayable_paths(
                        g, seq_length)

                    n_seqss_replayable.append(len(seqs_replayable))
                    n_seqss_non_replayable.append(len(seqs_non_replayable))


                # store graph, parameters, and sequence counts in database

                rcn = _models.ReplayCapacityNetwork(
                    graph_type=binary_graphs.erdos_renyi.__name__,
                    creation_parameters={'n_nodes': n_nodes, 'p_connect': p_connect},
                    networkx_adjacency_matrix=g.adj,
                    sequence_lengths=SEQ_LENGTHS,
                    n_sequencess_replayable=n_seqss_replayable,
                    n_sequencess_non_replayable=n_seqss_non_replayable)

                session.add(rcn)
                session.commit()


def test_er_capacity(test_log_filename):

    # set up testing

    test_engine, engine = start_database_modification_test(
        func=er_capacity, test_log_filename=test_log_filename)

    # use the models to build the schema in the test database

    _models.Base.metadata.create_all(test_engine)

    # open session for testing

    test_session = create_session(test_engine)

    # test function

    CONFIG = {
        'N_NODESS': [10, 12, 14],
        'P_CONNECTS': [.1, .2],
        'N_NETWORKSS': [2, 3],
        'SEQ_LENGTHS': [3, 4],
    }

    er_capacity(session=test_session, **CONFIG)

    test_engine.dispose()

    logging.info('Test completed.\n')