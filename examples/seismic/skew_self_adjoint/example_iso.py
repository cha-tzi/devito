import numpy as np

from devito import info
from examples.seismic import RickerSource
from examples.seismic.skew_self_adjoint import (default_setup_iso,
                                                SsaIsoAcousticWaveSolver)

shape = (221, 211, 201)
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 500.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

b, v, time_axis, src_coords, rec_coords = default_setup_iso(npad, shape, dtype, tmax=tmax)

solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                  src_coords, rec_coords, time_axis, space_order=8)

src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]

ns = src_coords.shape[0]
nr = rec_coords.shape[0]
info(time_axis)
info("ns, nr;         ", ns, nr)
info("grid.shape;     ", v.grid.shape)
info("b.shape;        ", b.grid.shape)
info("v.shape;        ", v.grid.shape)
info("grid.origin;    ", (v.grid.origin[0].data, v.grid.origin[1].data))
info("grid.spacing;   ", v.grid.spacing)

rec, _, _ = solver.forward(src)
