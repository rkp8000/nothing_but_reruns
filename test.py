from __future__ import division, print_function
import networkx as nx
import numpy as np
from scipy import stats
import unittest

import connectivity
import metrics
import network


class ConnectivityTestCase(unittest.TestCase):

    def test_example_feed_forward_grid_is_created_correctly(self):

        # FEED FORWARD WITH NO LATERAL SPREAD
        shape = (5, 3)
        spread = 1

        w_correct = np.array([
           # 00 01 02 10 11 12 20 21 22 30 31 32 40 41 42
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 00
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 01
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 02
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 22
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 30
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 31
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 32
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 40
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 41
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 42
           # 00 01 02 10 11 12 20 21 22 30 31 32 40 41 42
        ])

        w = connectivity.feed_forward_grid(shape=shape, spread=spread)
        np.testing.assert_array_equal(w, w_correct)

        # FEED FORWARD WITH 2 UNITS OF LATERAL SPREAD
        shape = (5, 3)
        spread = 2

        w_correct = np.array([
           # 00 01 02 10 11 12 20 21 22 30 31 32 40 41 42
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 00
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 01
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 02
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 12
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 21
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 22
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 30
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # 31
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # 32
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 40
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 41
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # 42
           # 00 01 02 10 11 12 20 21 22 30 31 32 40 41 42
        ])

        w = connectivity.feed_forward_grid(shape=shape, spread=spread)
        np.testing.assert_array_equal(w, w_correct)


class NetworkProbabilityCalculationTestCase(unittest.TestCase):

    def test_sequence_probability_is_calculated_correctly_in_softmax_wta_lingering_hyperexc_network(self):

        # NETWORK WITH SIMPLE WEIGHT MATRIX

        w = np.array([
            [0, 0, 0, 2, 0],
            [2, 0, 0, 2, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [4, 3, 2, 0, 0],
        ])

        g_w = 1.5
        g_x = 2.2
        g_d = 0.8
        t_x = 100

        drives = np.array([
            [2,   0, 0,   0, 0],
            [0, 1.5, 0,   0, 0],
            [0,   0, 0, 1.8, 0],
        ])

        seq = np.array([0, 1, 3])
        r_0 = np.array([0, 0, 0, 0, 0])
        xc_0 = np.array([0, 20, 0, 0, 0])

        p_seq_correct = \
            0.29173159541542848 * \
            0.59219297641971802 * \
            0.036145519643313952

        ntwk = network.SoftmaxWTAWithLingeringHyperexcitability(w, g_w, g_x, g_d, t_x)
        p_seq = ntwk.sequence_probability(seq, r_0, xc_0, drives)

        self.assertAlmostEqual(p_seq, p_seq_correct)


class MetricsTestCase(unittest.TestCase):

    def test_paths_are_correctly_found_in_example_network(self):

        a = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0]
        ])

        paths_3_correct = {
            (0, 2, 5),
            (1, 3, 5),
            (2, 5, 0),
            (2, 5, 4),
            (3, 5, 0),
            (3, 5, 4),
            (5, 0, 2),
        }

        paths_4_correct = {
            (0, 2, 5, 0),
            (0, 2, 5, 4),
            (1, 3, 5, 0),
            (1, 3, 5, 4),
            (2, 5, 0, 2),
            (3, 5, 0, 2),
            (5, 0, 2, 5),
        }

        g = nx.from_numpy_matrix(a.T, create_using=nx.DiGraph())

        paths_3 = set(metrics.paths_of_length(g, 3))
        paths_4 = set(metrics.paths_of_length(g, 4))

        self.assertEqual(paths_3, paths_3_correct)
        self.assertEqual(paths_4, paths_4_correct)

    def test_replayability_is_calculated_correctly_for_examples(self):

        a = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 0]
        ])

        replayable_paths = [
            (0, 2, 4, 3),
            (0, 2, 5, 0),
            (1, 3, 5, 0),
        ]

        non_replayable_paths = [
            (2, 4, 5, 0),
            (2, 5, 4, 3),
            (2, 4, 3, 5),
        ]

        for path in replayable_paths:

            self.assertTrue(metrics.path_is_replayable(path, a))

        for path in non_replayable_paths:

            self.assertFalse(metrics.path_is_replayable(path, a))

    def test_state_count_gives_correct_example_results(self):

        seq = [0, 0.5, 0.6, 0, 5, 0.5, 0, 0.6, 0.5, 0.6, 0.6, 8]

        states = [0, 0.5, 0.6, 5, 8]

        counts_correct = np.array([3, 3, 4, 1, 1], dtype=float)

        counts = metrics.occurrence_count(seq, states)

        np.testing.assert_array_almost_equal(counts, counts_correct)

    def test_transition_count_gives_correct_example_results(self):

        seq = [0, 1, 3, 2, 4, 3, 1, 2, 0, 3, 0, 2, 0, 4, 1, 2, 1, 2, 1, 1]

        states = np.arange(5)

        transitions_correct = np.array([
            [0, 0, 2, 1, 0],
            [1, 1, 2, 1, 1],
            [1, 3, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ], dtype=float)

        transitions = metrics.transition_count(seq, states)

        np.testing.assert_array_almost_equal(transitions, transitions_correct)

    def test_transition_matrix_DKL_gives_correct_example_results(self):

        # define transition matrices and stationary distributions

        x = np.array([
            [0.1, 0.5],
            [0.9, 0.5],
        ])

        x_0 = np.array([0.48564293, 0.87415728])

        y = np.array([
            [0.2, 0.3],
            [0.8, 0.7],
        ])

        y_0 = np.array([0.35112344, 0.93632918])

        # create joint matrices

        x_joint = x.copy()
        x_joint[:, 0] *= x_0
        x_joint[:, 1] *= x_0

        y_joint = y.copy()
        y_joint[:, 0] *= y_0
        y_joint[:, 1] *= y_0

        # make sure we're calculating the correct dkl between x and y

        dkl_correct = stats.entropy(x_joint.flatten(), y_joint.flatten(), base=2)

        dkl = metrics.transition_dkl(x, y, base=2)

        self.assertAlmostEqual(dkl, dkl_correct)

        # make sure that dkl between a distribution and itself is 0

        dkl_zero = metrics.transition_dkl(x_joint, x_joint, base=2)

        self.assertAlmostEqual(dkl_zero, 0)

    def test_gather_sequences_works_on_examples(self):

        x = [1, 2, 3, 4, 5, 6, 7]
        seq_len = 3

        seqs_correct = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
        ])

        seqs = metrics.gather_sequences(x, seq_len)
        np.testing.assert_array_equal(seqs, seqs_correct)

        x = [0, 2, 4, 6, 8, 11]
        seq_len = 4

        seqs_correct = np.array([
            [0, 2, 4, 6],
            [2, 4, 6, 8],
            [4, 6, 8, 11],
        ])

        seqs = metrics.gather_sequences(x, seq_len)
        np.testing.assert_array_equal(seqs, seqs_correct)

        x = [1, 3, 5, 7, 9, 11]
        seq_len = 1

        seqs_correct = np.array([
            [1],
            [3],
            [5],
            [7],
            [9],
            [11],
        ])

        seqs = metrics.gather_sequences(x, seq_len)
        np.testing.assert_array_equal(seqs, seqs_correct)

    def test_mutual_info_exact_works_with_example(self):

        # independent variables

        p_joint = np.array([
            [0.1, 0.3],
            [0.15, 0.45]
        ])

        self.assertAlmostEqual(metrics.mutual_info_exact(p_joint), 0)
        self.assertAlmostEqual(metrics.mutual_info_exact(p_joint.T), 0)

        # dependent variables

        p_joint = np.array([
            [0.4, 0.1],
            [0.1, 0.4],
        ])

        info_correct = 0.19274475702175742

        self.assertAlmostEqual(metrics.mutual_info_exact(p_joint), info_correct)
        self.assertAlmostEqual(metrics.mutual_info_exact(p_joint.T), info_correct)


if __name__ == '__main__':

    unittest.main()