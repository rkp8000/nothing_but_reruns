from __future__ import division, print_function
import numpy as np

def test_stdp_mediated_weight_changes_have_correct_sign_in_discrete_time_network():

    from network import BasicWithAthAndTwoLevelStdp as Network

    # make weight matrix
    w = 2 * np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ], dtype=float)

    # get masks for forward and backward connections
    f_mask = np.tril(w > 0)
    b_mask = np.triu(w > 0)

    # set stdp params
    stdp_params = {'w_1': 1, 'w_2': 3, 'alpha_1': .1, 'alpha_2': .15}

    # set up networks and drive paradigms for each test
    ntwks = [
        Network(th=2.5, w=w, g_x=0, t_x=0, rp=2, stdp_params=stdp_params),
        Network(th=1.5, w=w, g_x=0, t_x=0, rp=2, stdp_params=stdp_params),
        Network(th=2.5, w=w, g_x=0, t_x=0, rp=2, stdp_params=stdp_params),
    ]

    drives = [np.zeros((32, 6)) for _ in range(3)]

    for repeat in range(3):
        start = repeat*9 + 2

        # first drive paradigm has stim-activated one-direction sequence
        for drive_0_ctr in range(6):
            drives[0][start + drive_0_ctr, drive_0_ctr] = 5

        # second drive paradigm has internally generated one-direction sequence
        drives[1][start, 0] = 5

        # third drive paradigm has stim-activated alternating nonconsec
        for drive_2_ctr, node in enumerate([0, 2, 4, 1, 3, 5]):
            drives[2][start + drive_2_ctr, node] = 5

    # run networks
    r_0 = np.zeros((6,))
    xc_0 = np.zeros((6,))

    for ctr, (ntwk, drive) in enumerate(zip(ntwks, drives)):
        rs, xcs, ws = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drive, measure_w=lambda w:w)

        # make sure no nonzero weights have been added or removed
        assert len(ws) == len(drive)
        assert np.all(ws[-1].astype(bool) == w.astype(bool))

        # go through drive paradigm-/network-specific tests
        if ctr in [0, 1]:
            # ensure forward weights have increased
            assert np.all(ws[-1][f_mask] > w[f_mask])
            # ensure backward weights have decreased
            assert np.all(ws[-1][b_mask] < w[b_mask])
            # ensure forward increase was slightly larger than backwards decrease
            assert np.all(ws[-1][f_mask] - w[f_mask] > w[b_mask] - ws[-1][b_mask])

        else:
            # ensure that no weights have been changed
            assert np.all(ws[-1] == w)


def test_wta_network_correctly_calculates_node_distance_matrix_in_example_network():

    import networkx as nx
    from network import LocalWtaWithAthAndStdp

    # generate a random symmetric connected weight matrix
    np.random.seed(0)
    w = (np.random.rand(30, 30) < 0.4).astype(int)
    w = (w + w.T > 0).astype(int)

    dists = LocalWtaWithAthAndStdp(
        th=1, w=w, g_x=1, t_x=1, rp=2, stdp_params=None, wta_dist=2).node_distances

    # make sure networkx shortest path length function matches up with distance matrix
    for node_0, spls in nx.shortest_path_length(nx.Graph(w)).items():
        for node_1, spl in spls.items():
            assert dists[node_0, node_1] == dists[node_1, node_0] == spl

    # ensure distance values include 0s, 1s, and values greater than 1
    assert (dists == 0).sum() > 0
    assert (dists == 1).sum() > 0
    assert (dists > 1).sum() > 0


def test_wta_network_correctly_prevents_nodes_from_being_active_if_theyre_too_close_together():

    from itertools import combinations
    from connectivity import hexagonal_lattice
    from network import LocalWtaWithAthAndStdp

    # build hexagonal lattice connectivity matrix
    w_hex, nodes = hexagonal_lattice(4)

    # make network that should be always spontaneously active
    ntwk = LocalWtaWithAthAndStdp(
        th=-1, w=w_hex, g_x=0, t_x=0, rp=2, stdp_params=None, wta_dist=2)

    dists = ntwk.node_distances

    # run network for several time steps
    drives = np.zeros((20, len(nodes)))
    r_0 = np.zeros((len(nodes),))
    xc_0 = np.zeros((len(nodes),))
    rs, xcs = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives)

    # make sure at least some nodes are active at all time steps but there are no invalid
    # pairs of nodes
    for r in rs[1:]:
        assert r.sum() > 0

        for pair in combinations(r.nonzero()[0], 2):
            assert dists[pair[0], pair[1]] > 2

    # test for a network that is not spontaneously active
    ntwk = LocalWtaWithAthAndStdp(
        th=.5, w=w_hex, g_x=0, t_x=0, rp=2, stdp_params=None, wta_dist=2)

    # run network for several time steps
    drives = np.zeros((20, len(nodes)))
    drives[1, nodes.index((0, 0))] = 2
    rs, xcs = ntwk.run(r_0=r_0, xc_0=xc_0, drives=drives)

    for r in rs[1:]: assert r.sum() == 1