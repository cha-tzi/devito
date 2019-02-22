import numpy as np

from conftest import skipif
from devito import Grid, Constant, Function, TimeFunction, Eq, solve, Operator

pytestmark = skipif(['yask', 'ops'])


class TestConst(object):
    """
    Class for testing symbolic coefficients functionality
    """

    def test_const_change(self):
        """
        Test that the default replacement rules return the same
        as standard FD.
        """

        n = 5
        t = Constant(name='t', dtype=np.int32)

        grid = Grid(shape=(2, 2))
        x, y = grid.dimensions

        f = TimeFunction(name='f', grid=grid, save=n+1)
        f.data[:] = 0
        eq = Eq(f.dt-1)
        stencil = Eq(f.forward, solve(eq, f.forward))
        op = Operator([stencil])
        op.apply(time_m=0, time_M=n-1, dt=1)

        check = Function(name='check', grid=grid)
        eq_test = Eq(check, f[t, x, y])
        op_test = Operator([eq_test])
        for j in range(0, n+1):
            t.data = j  # Ensure constant is being updated correctly
            op_test.apply(t=t)
            assert(np.amax(check.data[:], axis=None) == j)
            assert(np.amin(check.data[:], axis=None) == j)
