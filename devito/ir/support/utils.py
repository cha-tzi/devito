from collections import OrderedDict, defaultdict

from devito.dimension import Dimension
from devito.ir.support.basic import Access
from devito.ir.support.space import Interval, Backward, Forward, Any
from devito.ir.support.stencil import Stencil
from devito.symbolics import retrieve_indexed
from devito.tools import as_tuple, flatten

__all__ = ['detect_intervals', 'detect_directions']


def detect_intervals(expr):
    """Return an iterable of :class:`Interval`s representing the data items
    accessed by the :class:`sympy.Eq` ``expr``."""
    # Deep retrieval of indexed objects in /expr/
    indexeds = retrieve_indexed(expr, mode='all')
    indexeds += flatten([retrieve_indexed(i) for i in e.indices] for e in indexeds)

    # Detect the indexeds' offsets along each dimension
    stencil = Stencil()
    for e in indexeds:
        for a in e.indices:
            if isinstance(a, Dimension):
                stencil[a].update([0])
            d = None
            off = [0]
            for i in a.args:
                if isinstance(i, Dimension):
                    d = i
                elif i.is_integer:
                    off += [int(i)]
            if d is not None:
                stencil[d].update(off)

    # Determine intervals and their iterators
    iterators = OrderedDict()
    for i in stencil.dimensions:
        if i.is_NonlinearDerived:
            iterators.setdefault(i.parent, []).append(stencil.entry(i))
        else:
            iterators.setdefault(i, [])
    intervals = []
    for k, v in iterators.items():
        offs = set.union(set(stencil.get(k)), *[i.ofs for i in v])
        intervals.append(Interval(k, min(offs), max(offs)))

    return intervals, iterators


def detect_directions(exprs):
    """Return a mapper from :class:`Dimension`s to iterables of
    :class:`IterationDirection`s for ``expr``."""
    exprs = as_tuple(exprs)

    writes = [Access(i.lhs, 'W') for i in exprs]
    reads = flatten(retrieve_indexed(i.rhs, mode='all') for i in exprs)
    reads = [Access(i, 'R') for i in reads]

    # Determine indexed-wise direction by looking at the vector distance
    mapper = defaultdict(set)
    for w in writes:
        for r in reads:
            if r.name != w.name:
                continue
            dimensions = [d for d in w.aindices if d is not None]
            for d in dimensions:
                try:
                    if w.distance(r, d) > 0:
                        mapper[d].add(Forward)
                        break
                    elif w.distance(r, d) < 0:
                        mapper[d].add(Backward)
                        break
                    else:
                        mapper[d].add(Any)
                except TypeError:
                    # Nothing can be deduced
                    mapper[d].add(Any)
                    break
            # Remainder
            for d in dimensions[dimensions.index(d) + 1:]:
                mapper[d].add(Any)

    # Add in any encountered Dimension
    mapper.update({d: {Any} for d in flatten(i.aindices for i in reads + writes)
                   if d is not None and d not in mapper})

    # Add in stepping dimensions (just in case they haven't been detected yet)
    # note: stepping dimensions may force a direction on the parent
    assert all(v == {Any} or mapper.get(k.parent, v) in [v, {Any}]
               for k, v in mapper.items() if k.is_Stepping)
    mapper.update({k.parent: set(v) for k, v in mapper.items()
                   if k.is_Stepping and mapper.get(k.parent) == {Any}})

    # Add in derived dimensions parents
    mapper.update({k.parent: set(v) for k, v in mapper.items()
                   if k.is_Derived and k.parent not in mapper})

    return mapper
