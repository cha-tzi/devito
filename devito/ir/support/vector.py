from collections import OrderedDict

from sympy import Basic

from devito.tools import as_tuple, is_integer, memoized_meth
from devito.types import Dimension

__all__ = ['Vector', 'LabeledVector']


class Vector(tuple):

    """
    A representation of a vector in Z^n.

    The elements of a Vector can be integers or any SymPy expression.

    Notes
    -----
    1) Vector-scalar comparison
    If a comparison between a vector and a non-vector is attempted, then the
    non-vector is promoted to a vector; if this is not possible, an exception
    is raised. This is handy because it turns a vector-scalar comparison into
    a vector-vector comparison with the scalar broadcasted to all vector entries.
    For example: ::

        (3, 4, 5) > 4 => (3, 4, 5) > (4, 4, 4) => False

    2) Comparing Vector entries when these are SymPy expression
    When we compare two entries that are both generic SymPy expressions, it is
    generally not possible to determine the truth value of the relation. For
    example, the truth value of `3*i < 4*j` cannot be determined. In some cases,
    however, the comparison is feasible; for example, `i + 4 < i` should always
    return false. A sufficient condition for two Vectors to be comparable is that
    their pair-wise indices are affine functions of the same variables, with
    coefficient 1.
    """

    def __new__(cls, *items):
        if not all(is_integer(i) or isinstance(i, Basic) for i in items):
            raise TypeError("Illegal Vector element type")
        return super(Vector, cls).__new__(cls, items)

    def _asvector(relax=False):
        def __asvector(func):
            def wrapper(self, other):
                if not isinstance(other, Vector):
                    try:
                        other = Vector(*other)
                    except TypeError:
                        # Not iterable
                        other = Vector(*(as_tuple(other)*len(self)))
                if relax is False and len(self) != len(other):
                    raise TypeError("Cannot operate with Vectors of different rank")
                return func(self, other)
            return wrapper
        return __asvector

    @_asvector()
    def __add__(self, other):
        return Vector(*[i + j for i, j in zip(self, other)])

    @_asvector()
    def __radd__(self, other):
        return self + other

    @_asvector()
    def __sub__(self, other):
        return Vector(*[i - j for i, j in zip(self, other)])

    @_asvector()
    def __rsub__(self, other):
        return self - other

    @_asvector(relax=True)
    def __eq__(self, other):
        return super(Vector, self).__eq__(other)

    def __hash__(self):
        return super(Vector, self).__hash__()

    @_asvector(relax=True)
    def __ne__(self, other):
        return super(Vector, self).__ne__(other)

    @_asvector()
    def __lt__(self, other):
        # This might raise an exception if the distance between the i-th entry
        # of /self/ and /other/ isn't integer, but rather a generic function
        # (and thus not comparable to 0). However, the implementation is "smart",
        # in the sense that it will return as soon as the first two comparable
        # entries (i.e., such that their distance is a non-zero integer) are found
        for i in self.distance(other):
            try:
                val = int(i)
            except TypeError:
                raise TypeError("Cannot compare due to non-comparable index functions")
            if val < 0:
                return True
            elif val > 0:
                return False

    @_asvector()
    def __gt__(self, other):
        return other.__lt__(self)

    @_asvector()
    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    @_asvector()
    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, key):
        ret = super(Vector, self).__getitem__(key)
        return Vector(*ret) if isinstance(key, slice) else ret

    def __repr__(self):
        return "(%s)" % ','.join(str(i) for i in self)

    @property
    def rank(self):
        return len(self)

    @property
    def sum(self):
        return sum(self)

    @property
    def is_constant(self):
        return all(is_integer(i) for i in self)

    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        The distance is a reflexive, transitive, and anti-symmetric relation,
        which establishes a total ordering amongst Vectors.

        The distance is a function [Vector x Vector --> D]. D is a tuple of length
        equal to the Vector ``rank``. The i-th entry of D, D_i, indicates whether
        the i-th component of ``self``, self_i, precedes (< 0), equals (== 0), or
        succeeds (> 0) the i-th component of ``other``, other_i.

        In particular, the *absolute value* of D_i represents the number of
        integer points that exist between self_i and sink_i.

        Examples
        --------
                 | 3 |           | 1 |               |  2  |
        source = | 2 | ,  sink = | 4 | , distance => | -2  |
                 | 1 |           | 5 |               | -4  |

        There are 2, 2, and 4 points between [3-2], [2-4], and [1-5], respectively.
        """
        return self - other


class LabeledVector(Vector):

    """
    A Vector that associates a Dimension to each element.
    """

    def __new__(cls, items=None):
        try:
            labels, values = zip(*items)
        except (ValueError, TypeError):
            labels, values = (), ()
        if not all(isinstance(i, Dimension) for i in labels):
            raise ValueError("All labels must be of type Dimension, got [%s]"
                             % ','.join(i.__class__.__name__ for i in labels))
        obj = super(LabeledVector, cls).__new__(cls, *values)
        obj.labels = labels
        return obj

    @classmethod
    def transpose(cls, *vectors):
        """
        Transpose a matrix represented as an iterable of homogeneous LabeledVectors.
        """
        if len(vectors) == 0:
            return LabeledVector()
        if not all(isinstance(v, LabeledVector) for v in vectors):
            raise ValueError("All items must be of type LabeledVector, got [%s]"
                             % ','.join(i.__class__.__name__ for i in vectors))
        T = OrderedDict()
        for v in vectors:
            for l, i in zip(v.labels, v):
                T.setdefault(l, []).append(i)
        return tuple((l, Vector(*i)) for l, i in T.items())

    def __repr__(self):
        return "(%s)" % ','.join('%s:%s' % (l, i) for l, i in zip(self.labels, self))

    def __hash__(self):
        return hash((tuple(self), self.labels))

    def __eq__(self, other):
        if isinstance(other, LabeledVector) and self.labels != other.labels:
            raise TypeError("Cannot compare due to mismatching `labels`")
        return super(LabeledVector, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, LabeledVector) and self.labels != other.labels:
            raise TypeError("Cannot compare due to mismatching `labels`")
        return super(LabeledVector, self).__lt__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return super(LabeledVector, self).__getitem__(index)
        elif isinstance(index, Dimension):
            for d in index._defines:
                if d in self.labels:
                    i = self.labels.index(d)
                    return super(LabeledVector, self).__getitem__(i)
            return None
        else:
            raise TypeError("Indices must be integers, slices, or Dimensions, not %s"
                            % type(index))

    def fromlabel(self, label, v=None):
        return self[label] if label in self.labels else v

    @memoized_meth
    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : LabeledVector
            The LabeledVector from which the distance is computed.
        """
        if not isinstance(other, LabeledVector):
            raise TypeError("Cannot compute distance from obj of type %s", type(other))
        if self.labels != other.labels:
            raise TypeError("Cannot compute distance due to mismatching `labels`")
        return LabeledVector(list(zip(self.labels, self - other)))
