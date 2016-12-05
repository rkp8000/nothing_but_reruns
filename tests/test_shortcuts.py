from __future__ import division, print_function


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

    assert zip_cproduct(z=z, c=c, **kwargs) == correct
