# convolution MFE
import numpy as np
from devito import Operator, Function, Grid,\
dimensions, Eq, Inc, ConditionalDimension, Dimension, TimeFunction

input_size = (2, 4, 3, 3)
a, b, c, d = dimensions('a b c d')
gridB = Grid(shape=(input_size[0], input_size[1],
                            input_size[2], input_size[3]),
                     dimensions=(a, b, c, d))
B = Function(name='B', grid=gridB, space_order=2,
                dtype=np.float64)
print("B:", B.shape)
print("B data:", B.data_with_halo.shape)

"""
B: (2, 4, 3, 3)
B data: (2, 4, 3, 3)
"""