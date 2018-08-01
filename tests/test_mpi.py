import numpy as np

import pytest
from conftest import skipif_yask

from devito import Grid, Function, TimeFunction, Eq, Operator
from devito.mpi import copy, sendrecv, update_halo
from devito.parameters import configuration
from devito.types import LEFT, RIGHT


def setup_module(module):
    configuration['mpi'] = True


def teardown_module(module):
    configuration['mpi'] = False


@skipif_yask
@pytest.mark.parallel(nprocs=[2, 4])
def test_basic_partitioning():
    grid = Grid(shape=(15, 15))
    f = Function(name='f', grid=grid)

    distributor = grid.distributor
    expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
        2: [(15, 8), (15, 7)],
        4: [(8, 8), (8, 7), (7, 8), (7, 7)]
    }
    assert f.shape == expected[distributor.nprocs][distributor.myrank]


@skipif_yask
@pytest.mark.parallel(nprocs=9)
def test_neighborhood_2d():
    grid = Grid(shape=(3, 3))
    x, y = grid.dimensions

    distributor = grid.distributor
    # Rank map:
    # ---------------y
    # | 0 | 1 | 2 |
    # -------------
    # | 3 | 4 | 5 |
    # -------------
    # | 6 | 7 | 8 |
    # -------------
    # |
    # x
    expected = {
        0: {x: {LEFT: None, RIGHT: 3}, y: {LEFT: None, RIGHT: 1}},
        1: {x: {LEFT: None, RIGHT: 4}, y: {LEFT: 0, RIGHT: 2}},
        2: {x: {LEFT: None, RIGHT: 5}, y: {LEFT: 1, RIGHT: None}},
        3: {x: {LEFT: 0, RIGHT: 6}, y: {LEFT: None, RIGHT: 4}},
        4: {x: {LEFT: 1, RIGHT: 7}, y: {LEFT: 3, RIGHT: 5}},
        5: {x: {LEFT: 2, RIGHT: 8}, y: {LEFT: 4, RIGHT: None}},
        6: {x: {LEFT: 3, RIGHT: None}, y: {LEFT: None, RIGHT: 7}},
        7: {x: {LEFT: 4, RIGHT: None}, y: {LEFT: 6, RIGHT: 8}},
        8: {x: {LEFT: 5, RIGHT: None}, y: {LEFT: 7, RIGHT: None}},
    }
    assert expected[distributor.myrank] == distributor.neighbours


@skipif_yask
@pytest.mark.parallel(nprocs=2)
def test_halo_exchange_bilateral():
    """
    Test halo exchange between two processes organised in a 1x2 cartesian grid.

    The initial ``data_with_halo`` looks like:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 0 0 0 0 0     0 0 0 0 0 0

    After the halo exchange, the following is expected and tested for:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 0 0 0 0 0     0 0 0 0 0 0
    """
    grid = Grid(shape=(12, 12))
    f = Function(name='f', grid=grid)

    distributor = grid.distributor
    f.data[:] = distributor.myrank + 1

    # Now trigger a halo exchange...
    f.data_with_halo   # noqa

    if distributor.myrank == 0:
        assert np.all(f.data_ro_with_halo[1:-1, -1] == 2.)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
    else:
        assert np.all(f.data_ro_with_halo[1:-1, 0] == 1.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
    assert np.all(f.data_ro_with_halo[0] == 0.)
    assert np.all(f.data_ro_with_halo[-1] == 0.)


@skipif_yask
@pytest.mark.parallel(nprocs=2)
def test_halo_exchange_bilateral_asymmetric():
    """
    Test halo exchange between two processes organised in a 1x2 cartesian grid.

    In this test, the size of left and right halo regions are different.

    The initial ``data_with_halo`` looks like:

           rank0           rank1
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 1 1 1 1 0     0 0 2 2 2 2 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0

    After the halo exchange, the following is expected and tested for:

           rank0           rank1
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 1 1 1 1 2     1 1 2 2 2 2 0
        0 0 0 0 0 0 0     0 0 0 0 0 0 0
    """
    grid = Grid(shape=(12, 12))
    f = Function(name='f', grid=grid, space_order=(1, 2, 1))

    distributor = grid.distributor
    f.data[:] = distributor.myrank + 1

    # Now trigger a halo exchange...
    f.data_with_halo   # noqa

    if distributor.myrank == 0:
        assert np.all(f.data_ro_with_halo[2:-1, -1] == 2.)
        assert np.all(f.data_ro_with_halo[:, 0:2] == 0.)
    else:
        assert np.all(f.data_ro_with_halo[2:-1, 0:2] == 1.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
    assert np.all(f.data_ro_with_halo[0:2] == 0.)
    assert np.all(f.data_ro_with_halo[-1] == 0.)


@skipif_yask
@pytest.mark.parallel(nprocs=4)
def test_halo_exchange_quadrilateral():
    """
    Test halo exchange between four processes organised in a 2x2 cartesian grid.

    The initial ``data_with_halo`` looks like:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 1 1 1 1 0     0 2 2 2 2 0
        0 0 0 0 0 0     0 0 0 0 0 0

           rank2           rank3
        0 0 0 0 0 0     0 0 0 0 0 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 3 3 3 3 0     0 4 4 4 4 0
        0 0 0 0 0 0     0 0 0 0 0 0

    After the halo exchange, the following is expected and tested for:

           rank0           rank1
        0 0 0 0 0 0     0 0 0 0 0 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 1 1 1 1 2     1 2 2 2 2 0
        0 3 3 3 3 4     3 4 4 4 4 0

           rank2           rank3
        0 1 1 1 1 2     1 2 2 2 2 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 3 3 3 3 4     3 4 4 4 4 0
        0 0 0 0 0 0     0 0 0 0 0 0
    """
    grid = Grid(shape=(12, 12))
    f = Function(name='f', grid=grid)

    distributor = grid.distributor
    f.data[:] = distributor.myrank + 1

    # Now trigger a halo exchange...
    f.data_with_halo   # noqa

    if distributor.myrank == 0:
        assert np.all(f.data_ro_with_halo[0] == 0.)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, -1] == 2.)
        assert np.all(f.data_ro_with_halo[-1, 1:-1] == 3.)
        assert f.data_ro_with_halo[-1, -1] == 4.
    elif distributor.myrank == 1:
        assert np.all(f.data_ro_with_halo[0] == 0.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, 0] == 1.)
        assert np.all(f.data_ro_with_halo[-1, 1:-1] == 4.)
        assert f.data_ro_with_halo[-1, 0] == 3.
    elif distributor.myrank == 2:
        assert np.all(f.data_ro_with_halo[-1] == 0.)
        assert np.all(f.data_ro_with_halo[:, 0] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, -1] == 4.)
        assert np.all(f.data_ro_with_halo[0, 1:-1] == 1.)
        assert f.data_ro_with_halo[0, -1] == 2.
    else:
        assert np.all(f.data_ro_with_halo[-1] == 0.)
        assert np.all(f.data_ro_with_halo[:, -1] == 0.)
        assert np.all(f.data_ro_with_halo[1:-1, 0] == 3.)
        assert np.all(f.data_ro_with_halo[0, 1:-1] == 2.)
        assert f.data_ro_with_halo[0, 0] == 1.


@skipif_yask
@pytest.mark.parallel(nprocs=[2, 4])
def test_ctypes_neighboors():
    grid = Grid(shape=(4, 4))
    distributor = grid.distributor

    attrs = ['xleft', 'xright', 'yleft', 'yright']
    expected = {  # nprocs -> [(rank0 xleft xright ...), (rank1 xleft ...), ...]
        2: [(-1, -1, -1, 1), (-1, -1, 0, -1)],
        4: [(-1, 2, -1, 1), (-1, 3, 0, -1), (0, -1, -1, 3), (1, -1, 2, -1)]
    }

    mapper = dict(zip(attrs, expected[distributor.nprocs][distributor.myrank]))
    _, _, obj = distributor._C_neighbours
    assert all(getattr(obj.value, k) == v for k, v in mapper.items())


@skipif_yask
class TestCodeGeneration(object):

    def test_iet_copy(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = copy(f, [t])
        assert str(iet.parameters) == """\
(buf(buf_x, buf_y), buf_x_size, buf_y_size, dat(dat_time, dat_x, dat_y),\
 dat_time_size, dat_x_size, dat_y_size, otime, ox, oy)"""
        assert """\
  for (int x = 0; x <= buf_x_size; x += 1)
  {
    for (int y = 0; y <= buf_y_size; y += 1)
    {
      buf[x][y] = dat[otime][x + ox][y + oy];
    }
  }""" in str(iet)

    def test_iet_sendrecv(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = sendrecv(f, [t])
        assert str(iet.parameters) == """\
(dat(dat_time, dat_x, dat_y), dat_time_size, dat_x_size, dat_y_size,\
 buf_x_size, buf_y_size, ogtime, ogx, ogy, ostime, osx, osy, fromrank, torank, comm)"""
        assert str(iet.body[0]) == """\
float (*restrict dat)[dat_x_size][dat_y_size] __attribute__((aligned(64))) =\
 (float (*)[dat_x_size][dat_y_size]) dat_vec;
float bufs[buf_x_size][buf_y_size] __attribute__((aligned(64)));
MPI_Request rrecv;
float bufg[buf_x_size][buf_y_size] __attribute__((aligned(64)));
MPI_Request rsend;
MPI_Irecv((float*)bufs,buf_x_size*buf_y_size,MPI_FLOAT,fromrank,\
MPI_ANY_TAG,comm,&rrecv);
gather_f((float*)bufg,buf_x_size,buf_y_size,(float*)dat,dat_time_size,dat_x_size,\
dat_y_size,ogtime,ogx,ogy);
MPI_Isend((float*)bufg,buf_x_size*buf_y_size,MPI_FLOAT,torank,\
MPI_ANY_TAG,comm,&rsend);
MPI_Wait(&rrecv,MPI_STATUS_IGNORE);
MPI_Wait(&rsend,MPI_STATUS_IGNORE);
scatter_f((float*)bufs,buf_x_size,buf_y_size,(float*)dat,dat_time_size,dat_x_size,\
dat_y_size,ostime,osx,osy);"""

    def test_iet_update_halo(self):
        grid = Grid(shape=(4, 4))
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        iet = update_halo(f, [t])
        assert str(iet.parameters) == """\
(f(t, x, y), mxl, mxr, myl, myr, comm, nb, otime, time_size, x_size, y_size)"""
        assert """\
MPI_Comm *comm = (MPI_Comm*) _comm;
struct neighbours *nb = (struct neighbours*) _nb;
if (mxl)
{
  sendrecv(f_vec,time_size,x_size + 1 + 1,y_size + 1 + 1,1,y_size + 1 + 1,\
otime,1,0,otime,x_size + 1,0,nb->xright,nb->xleft,comm);
}
if (mxr)
{
  sendrecv(f_vec,time_size,x_size + 1 + 1,y_size + 1 + 1,1,y_size + 1 + 1,\
otime,x_size,0,otime,0,0,nb->xleft,nb->xright,comm);
}
if (myl)
{
  sendrecv(f_vec,time_size,x_size + 1 + 1,y_size + 1 + 1,x_size + 1 + 1,1,\
otime,0,1,otime,0,y_size + 1,nb->yright,nb->yleft,comm);
}
if (myr)
{
  sendrecv(f_vec,time_size,x_size + 1 + 1,y_size + 1 + 1,x_size + 1 + 1,1,\
otime,0,y_size,otime,0,0,nb->yleft,nb->yright,comm);
}"""


@skipif_yask
def test_iet_simple_operator():
    grid = Grid(shape=(10,))
    x = grid.dimensions[0]
    t = grid.stepping_dim

    f = TimeFunction(name='f', grid=grid)
    f.data[:] = 0.

    op = Operator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
    print(op)
    from IPython import embed; embed()
    op.apply(time=1)
    from IPython import embed; embed()


if __name__ == "__main__":
    test_ctypes_neighboors()
