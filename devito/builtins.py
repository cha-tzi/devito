"""
Built-in Operators provided by Devito.
"""

from sympy import Abs, Pow
import numpy as np

import devito as dv

__all__ = ['assign', 'smooth', 'gaussian_smooth', 'initialize_function', 'norm',
           'sumall', 'inner', 'mmin', 'mmax']


def assign(f, v=0):
    """
    Assign a value to a Function.

    Parameters
    ----------
    f : Function
        The left-hand side of the assignment.
    v : scalar, optional
        The right-hand side of the assignment.
    """
    dv.Operator(dv.Eq(f, v), name='assign')()


def smooth(f, g, axis=None):
    """
    Smooth a Function through simple moving average.

    Parameters
    ----------
    f : Function
        The left-hand side of the smoothing kernel, that is the smoothed Function.
    g : Function
        The right-hand side of the smoothing kernel, that is the Function being smoothed.
    axis : Dimension or list of Dimensions, optional
        The Dimension along which the smoothing operation is performed. Defaults
        to ``f``'s innermost Dimension.

    Notes
    -----
    More info about simple moving average available at: ::

        https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """
    if g.is_Constant:
        # Return a scaled version of the input if it's a Constant
        f.data[:] = .9 * g.data
    else:
        if axis is None:
            axis = g.dimensions[-1]
        dv.Operator(dv.Eq(f, g.avg(dims=axis)), name='smoother')()


def gaussian_smooth(f, sigma=1, _order=4, mode='reflect'):
    """
    Gaussian smooth function.
    """
    class ObjectiveDomain(dv.SubDomain):

        name = 'objective_domain'

        def __init__(self, lw):
            super(ObjectiveDomain, self).__init__()
            self.lw = lw

        def define(self, dimensions):
            return {d: ('middle', self.lw, self.lw) for d in dimensions}

    def smooth(fo, fi, data):
        # TODO: Merge part of this with part of initialize_function and assign
        slices = tuple([slice(lw, -lw) for _ in range(fi.grid.dim)])
        fi.data[slices] = data
        eqs = []
        for d in fi.dimensions:
            dim_l = dv.SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                         thickness=lw)
            dim_r = dv.SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                          thickness=lw)
            if mode == 'constant':
                subsl = lw
                subsr = d.symbolic_max - lw
            elif mode == 'reflect':
                subsl = 2*lw - 1 - dim_l
                subsr = 2*(d.symbolic_max - lw) + 1 - dim_r
            else:
                raise ValueError("Mode not available")
            eqs += [dv.Eq(fi.subs({d: dim_l}), fi.subs({d: subsl}))]
            eqs += [dv.Eq(fi.subs({d: dim_r}), fi.subs({d: subsr}))]

            rhs = dv.generic_derivative(fi, d, 2*lw, 1)
            coeffs = dv.Coefficient(1, fi, d, weights)
            eqs += [dv.Eq(fo, rhs, coefficients=dv.Substitutions(coeffs),
                          subdomain=grid.subdomains['objective_domain'])]

            eqs += [dv.Eq(fi, fo, subdomain=grid.subdomains['objective_domain'])]

        dv.Operator(eqs)()

    def fset(f, g):
        indices = [slice(lw, -lw, 1) for _ in g.grid.dimensions]
        slices = (slice(None, None, 1), )*len(g.grid.dimensions)
        if isinstance(f, np.ndarray):
            f[slices] = g.data[tuple(indices)]
        elif isinstance(f, dv.Function):
            f.data[slices] = g.data[tuple(indices)]
        else:
            raise NotImplementedError

    lw = int(_order*sigma + 0.5)

    # Create the padded grid:
    objective_domain = ObjectiveDomain(lw)
    try:
        shape_padded = np.array(f.grid.shape) + 2*lw
    except AttributeError:
        shape_padded = np.array(f.shape) + 2*lw
    grid = dv.Grid(shape=shape_padded, subdomains=objective_domain)

    f_c = dv.Function(name='f_c', grid=grid, space_order=2*lw,
                      coefficients='symbolic', dtype=np.int32)
    f_o = dv.Function(name='f_o', grid=grid, coefficients='symbolic', dtype=np.int32)

    weights = np.exp(-0.5/sigma**2*(np.linspace(-lw, lw, 2*lw+1))**2)
    weights = weights/weights.sum()

    if isinstance(f, dv.Function):
        smooth(f_o, f_c, f.data[:])
    else:
        smooth(f_o, f_c, f)

    fset(f, f_o)

    return f


def initialize_function(function, data, nbpml, mode='constant'):
    """
    Initialize a `Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this function.

    Parameters
    ----------
    function : Function
        The initialised object.
    data : ndarray
        The data array used for initialisation.
    nbpml : int
        Number of PML layers for boundary damping.
    mode : str
        The function initialisation mode. 'constant' and 'reflect' are
        accepted.
    """
    slices = tuple([slice(nbpml, -nbpml) for _ in range(function.grid.dim)])
    function.data[slices] = data
    eqs = []

    for d in function.dimensions:
        dim_l = dv.SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                     thickness=nbpml)
        dim_r = dv.SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                      thickness=nbpml)
        if mode == 'constant':
            subsl = nbpml
            subsr = d.symbolic_max - nbpml
        elif mode == 'reflect':
            subsl = 2*nbpml - 1 - dim_l
            subsr = 2*(d.symbolic_max - nbpml) + 1 - dim_r
        else:
            raise ValueError("Mode not available")
        eqs += [dv.Eq(function.subs({d: dim_l}), function.subs({d: subsl}))]
        eqs += [dv.Eq(function.subs({d: dim_r}), function.subs({d: subsr}))]

    # TODO: Figure out why yask doesn't like it with dse/dle
    dv.Operator(eqs, name='padfunc', dse='noop', dle='noop')()


# Reduction-inducing builtins

class MPIReduction(object):
    """
    A context manager to build MPI-aware reduction Operators.
    """

    def __init__(self, *functions, op=dv.mpi.MPI.SUM):
        grids = {f.grid for f in functions}
        if len(grids) == 0:
            self.grid = None
        elif len(grids) == 1:
            self.grid = grids.pop()
        else:
            raise ValueError("Multiple Grids found")
        dtype = {f.dtype for f in functions}
        if len(dtype) == 1:
            self.dtype = dtype.pop()
        else:
            raise ValueError("Illegal mixed data types")
        self.v = None
        self.op = op

    def __enter__(self):
        i = dv.Dimension(name='i',)
        self.n = dv.Function(name='n', shape=(1,), dimensions=(i,),
                             grid=self.grid, dtype=self.dtype)
        self.n.data[0] = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.grid is None or not dv.configuration['mpi']:
            assert self.n.data.size == 1
            self.v = self.n.data[0]
        else:
            comm = self.grid.distributor.comm
            self.v = comm.allreduce(np.asarray(self.n.data), self.op)[0]


@dv.switchconfig(log_level='ERROR')
def norm(f, order=2):
    """
    Compute the norm of a Function.

    Parameters
    ----------
    f : Function
        Input Function.
    order : int, optional
        The order of the norm. Defaults to 2.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    with MPIReduction(f) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], Abs(Pow(p, order)))],
                         name='norm%d' % order)
        op.apply(**kwargs)

    v = Pow(mr.v, 1/order)

    return np.float(v)


def sumall(f):
    """
    Compute the sum of all Function data.

    Parameters
    ----------
    f : Function
        Input Function.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    with MPIReduction(f) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], p)], name='sum')
        op.apply(**kwargs)

    return np.float(mr.v)


def inner(f, g):
    """
    Inner product of two Functions.

    Parameters
    ----------
    f : Function
        First input operand
    g : Function
        Second input operand

    Raises
    ------
    ValueError
        If the two input Functions are defined over different grids, or have
        different dimensionality, or their dimension-wise sizes don't match.
        If in input are two SparseFunctions and their coordinates don't match,
        the exception is raised.

    Notes
    -----
    The inner product is the sum of all dimension-wise products. For 1D Functions,
    the inner product corresponds to the dot product.
    """
    # Input check
    if f.is_TimeFunction and f._time_buffering != g._time_buffering:
        raise ValueError("Cannot compute `inner` between save/nosave TimeFunctions")
    if f.shape != g.shape:
        raise ValueError("`f` and `g` must have same shape")
    if f._data is None or g._data is None:
        raise ValueError("Uninitialized input")
    if f.is_SparseFunction and not np.all(f.coordinates_data == g.coordinates_data):
        raise ValueError("Non-matching coordinates")

    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    rhs, eqns = f.guard(f*g) if f.is_SparseFunction else (f*g, [])

    with MPIReduction(f, g) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], rhs)], name='inner')
        op.apply(**kwargs)

    return np.float(mr.v)


def mmin(f):
    """
    Retrieve the minimum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if isinstance(f, dv.Constant):
        return f.data
    elif isinstance(f, dv.Function):
        with MPIReduction(f, op=dv.mpi.MPI.MIN) as mr:
            mr.n.data[0] = np.min(f.data_ro_domain).item()
        return mr.v.item()
    else:
        raise ValueError("Expected Function, not `%s`" % type(f))


def mmax(f):
    """
    Retrieve the maximum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if isinstance(f, dv.Constant):
        return f.data
    elif isinstance(f, dv.Function):
        with MPIReduction(f, op=dv.mpi.MPI.MAX) as mr:
            mr.n.data[0] = np.max(f.data_ro_domain).item()
        return mr.v.item()
    else:
        raise ValueError("Expected Function, not `%s`" % type(f))
