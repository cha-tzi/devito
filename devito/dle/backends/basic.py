from collections import OrderedDict

import cgen as c
import numpy as np

from devito.dle.backends import AbstractRewriter, dle_pass, complang_ALL
from devito.ir.iet import (Denormals, Call, Callable, List, ArrayCast,
                           Transformer, FindSymbols, retrieve_iteration_tree,
                           filter_iterations, iet_insert_C_decls, derive_parameters)
from devito.parameters import configuration
from devito.symbolics import as_symbol
from devito.types import Scalar, IncrDimension


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._create_efuncs(state)

    @dle_pass
    def _avoid_denormals(self, nodes, state):
        """
        Introduce nodes in the Iteration/Expression tree that will expand to C
        macros telling the CPU to flush denormal numbers in hardware. Denormals
        are normally flushed when using SSE-based instruction sets, except when
        compiling shared objects.
        """
        return (List(body=(Denormals(), nodes)),
                {'includes': ('xmmintrin.h', 'pmmintrin.h')})

    @dle_pass
    def _create_efuncs(self, nodes, state):
        """
        Extract Iteration sub-trees and turn them into Calls+Callables.

        Currently, only tagged, elementizable Iteration objects are targeted.
        """
        noinline = self._compiler_decoration('noinline', c.Comment('noinline?'))

        efuncs = OrderedDict()
        mapper = {}
        for tree in retrieve_iteration_tree(nodes, mode='superset'):
            # Search an elementizable sub-tree (if any)
            tagged = filter_iterations(tree, lambda i: i.tag is not None, 'asap')
            if not tagged:
                continue
            root = tagged[0]
            if not root.is_Elementizable:
                continue
            target = tree[tree.index(root):]

            # Build a new Iteration/Expression tree with free bounds
            free = []
            defined_args = {}  # Map of argument values defined by loop bounds
            for i in target:
                name, bounds = i.dim.name, i.symbolic_bounds
                # Iteration bounds
                _min = Scalar(name='%sf_m' % name, dtype=np.int32)
                _max = Scalar(name='%sf_M' % name, dtype=np.int32)
                defined_args[_min.name] = bounds[0]
                defined_args[_max.name] = bounds[1]

                # Iteration unbounded indices
                ufunc = [Scalar(name='%s_ub%d' % (name, j), dtype=np.int32)
                         for j in range(len(i.uindices))]
                defined_args.update({uf.name: j.symbolic_min
                                     for uf, j in zip(ufunc, i.uindices)})
                limits = [Scalar(name=_min.name, dtype=np.int32),
                          Scalar(name=_max.name, dtype=np.int32), 1]
                uindices = [IncrDimension(j.parent, i.dim + as_symbol(k), 1, j.name)
                            for j, k in zip(i.uindices, ufunc)]
                free.append(i._rebuild(limits=limits, offsets=None, uindices=uindices))

            # Construct elemental function body
            free = Transformer(dict((zip(target, free))), nested=True).visit(root)
            items = FindSymbols().visit(free)

            # Insert array casts
            casts = [ArrayCast(i) for i in items if i.is_Tensor]
            free = List(body=casts + [free])

            # Insert declarations
            external = [i for i in items if i.is_Array]
            free = iet_insert_C_decls(free, external)

            # Derive efunc parameters and arguments
            args = []
            params = []
            for i in derive_parameters(free):
                if i.name in defined_args:
                    args.append(defined_args[i.name])
                    params.append(i)
                elif i.is_Dimension:
                    d = Scalar(name=i.name, dtype=i.dtype)
                    args.append(d)
                    params.append(d)
                else:
                    args.append(i)
                    params.append(i)

            # Create the Call
            name = "f_%d" % root.tag
            mapper[root] = List(header=noinline, body=Call(name, args))

            # Create the Callable
            efuncs.setdefault(name, Callable(name, free, 'void', params, 'static'))

        # Transform the main tree
        processed = Transformer(mapper).visit(nodes)

        return processed, {'efuncs': efuncs.values()}

    def _compiler_decoration(self, name, default=None):
        key = configuration['compiler'].__class__.__name__
        complang = complang_ALL.get(key, {})
        return complang.get(name, default)
