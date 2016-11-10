from __future__ import division, print_function
import numpy as np


def test_hexagonal_lattice_is_created_correctly():

        import connectivity

        # make sure node set is made correctly
        d = 4
        nodes_correct = [
                        (-3, -3), (-3, -1), (-3, 1), (-3, 3),
                    (-2, -4), (-2, -2), (-2, 0), (-2, 2), (-2, 4),
                (-1, -5), (-1, -3), (-1, -1), (-1, 1), (-1, 3), (-1, 5),
            (0, -6), (0, -4), (0, -2), (0, 0), (0, 2), (0, 4), (0, 6),
                (1, -5), (1, -3), (1, -1), (1, 1), (1, 3), (1, 5),
                    (2, -4), (2, -2), (2, 0), (2, 2), (2, 4),
                        (3, -3), (3, -1), (3, 1), (3, 3),
        ]

        w, nodes = connectivity.hexagonal_lattice(d)

        assert nodes == nodes_correct
        assert np.all(w == w.T)

        d = 3
        nodes_correct = [
                    (-2, -2), (-2, 0), (-2, 2),
                (-1, -3), (-1, -1), (-1, 1), (-1, 3),
            (0, -4), (0, -2), (0, 0), (0, 2), (0, 4),
                (1, -3), (1, -1), (1, 1), (1, 3),
                    (2, -2), (2, 0), (2, 2),
        ]

        w, nodes = connectivity.hexagonal_lattice(d)

        assert nodes == nodes_correct
        assert np.all(w == w.T)

        # make sure connectivity is made correctly on small example

        d = 2
        nodes_correct = [
                (-1, -1), (-1, 1),
            (0, -2), (0, 0), (0, 2),
                (1, -1), (1, 1),
        ]

        w_correct = np.array([
           #-1 -1  0  0  0  1  1
           #-1  1 -2  0  2 -1  1
            [0, 1, 1, 1, 0, 0, 0],  # (-1, 1)
            [1, 0, 0, 1, 1, 0, 0],  # (-1, 3)
            [1, 0, 0, 1, 0, 1, 0],  # (0, -2)
            [1, 1, 1, 0, 1, 1, 1],  # (0, 0)
            [0, 1, 0, 1, 0, 0, 1],  # (0, 2)
            [0, 0, 1, 1, 0, 0, 1],  # (1, -1)
            [0, 0, 0, 1, 1, 1, 0],  # (1, 1)
        ])

        w, nodes = connectivity.hexagonal_lattice(d)

        assert nodes == nodes_correct
        assert np.all(w == w.T)
