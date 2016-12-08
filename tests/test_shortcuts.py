from __future__ import division, print_function


def test_reorder_indexes_works_on_examples():
    from shortcuts import reorder_idxs

    original = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    first = [(6, 6), (2, 2), (1, 1)]

    reordered_correct = [(6, 6), (2, 2), (1, 1), (0, 0), (3, 3), (4, 4), (5, 5)]
    idxs_correct = [6, 2, 1, 0, 3, 4, 5]

    reordered, idxs = reorder_idxs(original, first)

    assert reordered == reordered_correct
    assert idxs == idxs_correct


def test_zip_cproduct_works_on_examples():
    from shortcuts import zip_cproduct

    x = ['x1', 'x2', 'x3']
    y = ['y1', 'y2', 'y3']

    z0 = ['z00', 'z01']
    z1 = ['z10', 'z11']

    kwargs = {'x': x, 'y': y, 'z0': z0, 'z1': z1}
    z = ['x', 'y']
    c = ['z0', 'z1']

    correct = [
        ('x1', 'y1', 'z00', 'z10'),
        ('x1', 'y1', 'z00', 'z11'),
        ('x1', 'y1', 'z01', 'z10'),
        ('x1', 'y1', 'z01', 'z11'),
        ('x2', 'y2', 'z00', 'z10'),
        ('x2', 'y2', 'z00', 'z11'),
        ('x2', 'y2', 'z01', 'z10'),
        ('x2', 'y2', 'z01', 'z11'),
        ('x3', 'y3', 'z00', 'z10'),
        ('x3', 'y3', 'z00', 'z11'),
        ('x3', 'y3', 'z01', 'z10'),
        ('x3', 'y3', 'z01', 'z11'),
    ]

    assert zip_cproduct(
        z=z, c=c, order=['x', 'y', 'z0', 'z1'], kwargs=kwargs) == correct

    # reordered example

    correct = [
        ('y1', 'z10', 'x1', 'z00'),
        ('y1', 'z11', 'x1', 'z00'),
        ('y1', 'z10', 'x1', 'z01'),
        ('y1', 'z11', 'x1', 'z01'),
        ('y2', 'z10', 'x2', 'z00'),
        ('y2', 'z11', 'x2', 'z00'),
        ('y2', 'z10', 'x2', 'z01'),
        ('y2', 'z11', 'x2', 'z01'),
        ('y3', 'z10', 'x3', 'z00'),
        ('y3', 'z11', 'x3', 'z00'),
        ('y3', 'z10', 'x3', 'z01'),
        ('y3', 'z11', 'x3', 'z01'),
    ]

    assert zip_cproduct(
        z=z, c=c, order=['y', 'z1', 'x', 'z0'], kwargs=kwargs) == correct
