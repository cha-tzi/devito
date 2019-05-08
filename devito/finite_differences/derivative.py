import sympy

from collections import Iterable, OrderedDict
from devito.finite_differences.finite_difference import (generic_derivative,
                                                         first_derivative,
                                                         cross_derivative)
from devito.finite_differences.differentiable import Differentiable
from devito.finite_differences.tools import centered, direct, transpose, left, right
from devito.tools import as_tuple


class Derivative(sympy.Derivative, Differentiable):

    """
    An unevaluated Derivative, which carries metadata (Dimensions,
    derivative order, etc) describing how the derivative will be expanded
    upon evaluation.

    Parameters
    ----------
    expr : expr-like
        Expression for which the Derivative is produced.
    dims : Dimension or tuple of Dimension
        Dimenions w.r.t. which to differentiate.
    fd_order : int or tuple of int, optional
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil. Defaults to 1.
    deriv_order: int or tuple of int, optional
        Derivative order. Defaults to 1.
    stagger : Side or tuple of Side, optional
        Shift of the finite-difference approximation. Defaults to ``centered``.
    side : Side or tuple of Side, optional
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1). Defaults to ``centered``.
    transpose : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to ``direct``.

    Examples
    --------
    Creation

    >>> from devito import Function, Derivative, Grid
    >>> grid = Grid((10, 10))
    >>> x, y = grid.dimensions
    >>> u = Function(name="u", grid=grid, space_order=2)
    >>> Derivative(u, x)
    Derivative(u(x, y), x)

    This can also be obtained via the differential shortcut

    >>> u.dx
    Derivative(u(x, y), x)

    You can also specify the order as a keyword argument

    >>> Derivative(u, x, deriv_order=2)
    Derivative(u(x, y), (x, 2))

    Or as a tuple

    >>> Derivative(u, (x, 2))
    Derivative(u(x, y), (x, 2))

    Once again, this can be obtained via shortcut notation

    >>> u.dx2
    Derivative(u(x, y), (x, 2))
    """

    _state = ('expr', 'dims', 'side', 'stagger', 'fd_order', 'transpose')

    def __new__(cls, expr, *dims, **kwargs):
        # Check dims, can be a dimensions, multiple dimensions as a tuple
        # or a tuple of tuple (ie ((x,1),))
        if len(dims) == 1:
            if isinstance(dims[0], Iterable):
                orders = kwargs.get('deriv_order', dims[0][1])
                if dims[0][1] != orders:
                    raise ValueError("Two different value of deriv_order")
                new_dims = tuple([dims[0][0]]*dims[0][1])
            else:
                orders = kwargs.get('deriv_order', 1)
                new_dims = tuple([dims[0]]*orders)
        else:
            # ie ((x, 2), (y, 3))
            new_dims = []
            orders = []
            d_ord = kwargs.get('deriv_order', tuple([1]*len(dims)))
            for d, o in zip(dims, d_ord):
                if isinstance(d, Iterable):
                    new_dims += [d[0] for _ in range(d[1])]
                    orders += [d[1]]
                else:
                    new_dims += [d for _ in range(o)]
                    orders += [o]
            new_dims = as_tuple(new_dims)

        kwargs["evaluate"] = False
        kwargs["simplify"] = False
        obj = sympy.Derivative.__new__(cls, expr, *new_dims, **kwargs)
        obj._dims = tuple(OrderedDict.fromkeys(new_dims))
        obj._fd_order = kwargs.get('fd_order', 1)
        obj._deriv_order = orders
        obj._side = kwargs.get("side", None)
        obj._stagger = kwargs.get("stagger", tuple([centered]*len(obj._dims)))
        obj._transpose = kwargs.get("transpose", direct)

        return obj

    @property
    def dims(self):
        return self._dims

    @property
    def fd_order(self):
        return self._fd_order

    @property
    def deriv_order(self):
        return self._deriv_order

    @property
    def stagger(self):
        return self._stagger

    @property
    def side(self):
        return self._side

    @property
    def is_Staggered(self):
        return self.expr.is_Staggered

    @property
    def indices(self):
        return self.expr.indices

    @property
    def staggered(self):
        return self.expr.staggered

    @property
    def transpose(self):
        return self._transpose

    @property
    def T(self):
        """Transpose of the Derivative."""
        if self._transpose == direct:
            adjoint = transpose
        else:
            adjoint = direct

        return Derivative(self.expr, *self.dims, deriv_order=self.deriv_order,
                          fd_order=self.fd_order, side=self.side, stagger=self.stagger,
                          transpose=adjoint)

    @property
    def evaluate(self):
        expr = getattr(self.expr, 'evaluate', self.expr)
        if self.side in [left, right] and self.deriv_order == 1:
            res = first_derivative(expr, self.dims[0], self.fd_order,
                                   side=self.side, matvec=self.transpose)
        elif len(self.dims) > 1:
            res = cross_derivative(expr, self.dims, self.fd_order, self.deriv_order,
                                   matvec=self.transpose, stagger=self.stagger)
        else:
            res = generic_derivative(expr, *self.dims, self.fd_order,
                                     self.deriv_order, stagger=self.stagger,
                                     matvec=self.transpose)
        return res
