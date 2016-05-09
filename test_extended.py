from __future__ import division, print_function
import networkx as nx
import numpy as np
from scipy import stats
import unittest

import connectivity
import metrics
import network


class MetricsTestCase(unittest.TestCase):

    def test_mutual_info_monte_carlo_estimate_works_with_examples(self):

        mc_samples = [500, 500, 500, 500]

        # independent variables 1

        p_ab = np.array([
            [0.1, 0.3],
            [0.15, 0.45]
        ])
        p_a = p_ab.sum(1)

        def p_b_given_a(b, a):

            return (p_ab[a, :] / p_ab[a, :].sum())[b]

        def sample_a():

            return np.random.choice([0, 1], p=p_a)

        def sample_b_given_a(a):

            p = p_ab[a, :] / p_ab[a, :].sum()

            return np.random.choice([0, 1], p=p)

        ind_info_estimate = metrics.mutual_info_monte_carlo_estimate(
            sample_a=sample_a, sample_b_given_a=sample_b_given_a,
            p_b_given_a=p_b_given_a, mc_samples=mc_samples)

        print('ind info estimate: {}'.format(ind_info_estimate))
        print('ind info exact: {}'.format(metrics.mutual_info_exact(p_ab)))

        # weakly dependent variables

        p_ab = np.array([
            [0.4, 0.1],
            [0.1, 0.4]
        ])
        p_a = p_ab.sum(1)

        def p_b_given_a(b, a):

            return (p_ab[a, :] / p_ab[a, :].sum())[b]

        def sample_a():

            return np.random.choice([0, 1], p=p_a)

        def sample_b_given_a(a):

            p = p_ab[a, :] / p_ab[a, :].sum()

            return np.random.choice([0, 1], p=p)

        wk_dep_info_estimate = metrics.mutual_info_monte_carlo_estimate(
            sample_a=sample_a, sample_b_given_a=sample_b_given_a,
            p_b_given_a=p_b_given_a, mc_samples=mc_samples)

        print('wk dep info estimate: {}'.format(wk_dep_info_estimate))
        print('wk dep info exact: {}'.format(metrics.mutual_info_exact(p_ab)))

        # strongly dependent variables

        p_ab = np.array([
            [0.49, 0.01],
            [0.01, 0.49]
        ])
        p_a = p_ab.sum(1)

        def p_b_given_a(b, a):

            return (p_ab[a, :] / p_ab[a, :].sum())[b]

        def sample_a():

            return np.random.choice([0, 1], p=p_a)

        def sample_b_given_a(a):

            p = p_ab[a, :] / p_ab[a, :].sum()

            return np.random.choice([0, 1], p=p)

        stg_dep_info_estimate = metrics.mutual_info_monte_carlo_estimate(
            sample_a=sample_a, sample_b_given_a=sample_b_given_a,
            p_b_given_a=p_b_given_a, mc_samples=mc_samples)

        print('stg dep info estimate: {}'.format(stg_dep_info_estimate))
        print('stg dep info exact: {}'.format(metrics.mutual_info_exact(p_ab)))

        self.assertTrue(ind_info_estimate < wk_dep_info_estimate < stg_dep_info_estimate)


if __name__ == '__main__':

    unittest.main()