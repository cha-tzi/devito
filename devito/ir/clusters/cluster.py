from itertools import chain

import numpy as np
from cached_property import cached_property
from frozendict import frozendict

from devito.ir.equations import ClusterizedEq
from devito.ir.support import IterationSpace, DataSpace, Scope, detect_io
from devito.symbolics import estimate_cost, retrieve_indexed
from devito.tools import as_tuple

__all__ = ["Cluster", "ClusterGroup"]


class Cluster(object):

    """
    A Cluster is an ordered sequence of expressions in an IterationSpace.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        An ordered sequence of expressions computing a tensor.
    ispace : IterationSpace
        The cluster iteration space.
    dspace : DataSpace
        The cluster data space.
    atomics : list, optional
        Dimensions inducing a data dependence with other Clusters.
    guards : dict, optional
        Mapper from Dimensions to expr-like, representing the conditions under
        which the Cluster should be computed.
    """

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None):
        self._exprs = tuple(ClusterizedEq(i, ispace=ispace, dspace=dspace)
                            for i in as_tuple(exprs))
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = frozenset(atomics or [])
        self._guards = frozendict(guards or {})

    def __repr__(self):
        return "Cluster([%s])" % ('\n' + ' '*9).join('%s' % i for i in self.exprs)

    @classmethod
    def from_clusters(cls, *clusters):
        """
        Build a new Cluster from a sequence of pre-existing Clusters with
        compatible IterationSpace.
        """
        assert len(clusters) > 0
        root = clusters[0]
        assert all(root.ispace.is_compatible(c.ispace) for c in clusters)
        exprs = chain(*[c.exprs for c in clusters])
        ispace = IterationSpace.merge(*[c.ispace for c in clusters])
        dspace = DataSpace.merge(*[c.dspace for c in clusters])
        return Cluster(exprs, ispace, dspace)

    def rebuild(self, exprs):
        """
        Build a new Cluster from a sequence of expressions. All other attributes
        are inherited from ``self``.
        """
        return Cluster(exprs, self.ispace, self.dspace, self.atomics, self.guards)

    @property
    def exprs(self):
        return self._exprs

    @property
    def ispace(self):
        return self._ispace

    @property
    def itintervals(self):
        return self.ispace.itintervals

    @property
    def shape(self):
        return self.ispace.shape

    @property
    def dspace(self):
        return self._dspace

    @property
    def atomics(self):
        return self._atomics

    @property
    def guards(self):
        return self._guards

    @cached_property
    def functions(self):
        return set.union(*[set(i.dspace.parts) for i in self.exprs])

    @cached_property
    def scope(self):
        return Scope(self.exprs)

    @cached_property
    def is_sparse(self):
        return any(f.is_SparseFunction for f in self.functions)

    @cached_property
    def is_dense(self):
        return not self.is_sparse

    @cached_property
    def dtype(self):
        """
        The arithmetic data type of the Cluster. If the Cluster performs
        floating point arithmetic, then the expressions performing integer
        arithmetic are ignored, assuming that they are only carrying out array
        index calculations. If two expressions perform floating point
        calculations with mixed precision, the data type with highest precision
        is returned.
        """
        dtypes = {i.dtype for i in self.exprs}
        fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating)}
        if len(fdtypes) > 1:
            raise NotImplementedError("Unsupported Cluster with mixed floating "
                                      "point arithmetic %s" % str(fdtypes))
        elif len(fdtypes) == 1:
            return fdtypes.pop()
        elif len(dtypes) == 1:
            return dtypes.pop()
        else:
            raise ValueError("Unsupported Cluster [mixed integer arithmetic ?]")

    @cached_property
    def ops(self):
        """The Cluster operation count."""
        return self.ispace.size*sum(estimate_cost(i) for i in self.exprs)

    @cached_property
    def traffic(self):
        """
        The Cluster compulsary traffic (number of reads/writes), as a mapper
        from Functions to IntervalGroups.

        Notes
        -----
        If a Function is both read and written, then it is counted twice.
        """
        reads, writes = detect_io(self.exprs, relax=True)
        accesses = [(i, 'r') for i in reads] + [(i, 'w') for i in writes]
        ret = {}
        for i, mode in accesses:
            if not i.is_Tensor:
                continue
            elif i in self.dspace.parts:
                # Stencils extend the data spaces beyond the iteration spaces
                intervals = self.dspace.parts[i]
                # Assume that invariant dimensions always cause new loads/stores
                invariants = self.ispace.intervals.drop(intervals.dimensions)
                intervals = intervals.generate('union', invariants, intervals)
                ret[(i, mode)] = intervals
            else:
                ret[(i, mode)] = self.ispace.intervals
        return ret


class ClusterGroup(tuple):

    """An immutable iterable of Clusters."""

    def __new__(cls, items):
        assert all(isinstance(i, Cluster) for i in items)
        return super(ClusterGroup, cls).__new__(cls, items)

    @property
    def dspace(self):
        """Return the DataSpace of this ClusterGroup."""
        return DataSpace.merge(*[i.dspace for i in self])

    @property
    def dtype(self):
        """
        The arithmetic data type of this ClusterGroup. If at least one
        Cluster performs floating point arithmetic, then Clusters performing
        integer arithmetic are ignored. If two Clusters perform floating
        point calculations with different precision, return the data type with
        highest precision.
        """
        dtypes = {i.dtype for i in self}
        fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating)}
        if len(fdtypes) > 1:
            raise NotImplementedError("Unsupported ClusterGroup with mixed floating "
                                      "point arithmetic %s" % str(fdtypes))
        elif len(fdtypes) == 1:
            return fdtypes.pop()
        elif len(dtypes) == 1:
            return dtypes.pop()
        else:
            raise ValueError("Unsupported ClusterGroup [mixed integer arithmetic ?]")

    @property
    def meta(self):
        """
        Returns
        -------
        dtype, DSpace
            The data type and the data space of the ClusterGroup.
        """
        return (self.dtype, self.dspace)
