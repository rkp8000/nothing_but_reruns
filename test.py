from __future__ import division, print_function
import numpy as np
import unittest

import connectivity
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


if __name__ == '__main__':
    unittest.main()