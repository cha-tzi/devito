# convolution MFE
import numpy as np
from devito import Operator, Function, Grid,\
dimensions, Eq, Inc, Dimension, TimeFunction, SpaceDimension
dim_index = 0
def default_dim_allocator(count):
    global dim_index
    names = ''
    for i in range(count):
        names += 'd' + str(dim_index) + ' '
        dim_index += 1
    names = names[:-1]
    return Dimension(names)

def space_dim_allocator(count):
    global dim_index
    names = ''
    for i in range(count):
        names += 'd' + str(dim_index) + ' '
        dim_index += 1
    names = names[:-1]
    return SpaceDimension(names)


input_size = (2, 4, 3, 3)
a, b, c, d = dimensions('a b c d')
gridB = Grid(shape=(input_size[0], input_size[1],
                            input_size[2], input_size[3]),
                     dimensions=(a, b, c, d))
B = Function(name='B', grid=gridB, space_order=2,
                dtype=np.float64)
print("B:", B.shape)
print("B data:", B.data_with_halo.shape)

# gridC = Grid(shape=(input_size[0], input_size[2],
#                             input_size[3]))
# C = Function(name='C', grid=gridC, space_order=(0,0,2,2),
#                 dtype=np.float64)
# print("C:", C.shape)
# print("C data:", C.data_with_halo.shape)


e, f = dimensions('e  f')
# g = SpaceDimension('g')
# h = SpaceDimension('h')
# print("ef", e+f )
# print("g h ", g+h )
# print("e+g",e+g )
# print("", )
print("tuple", (default_dim_allocator(1), default_dim_allocator(1),
                space_dim_allocator(1), space_dim_allocator(1) ))
gridD = Grid(shape=(input_size[0], input_size[1],
                    input_size[2], input_size[3]),
                    dimensions=(default_dim_allocator(1), default_dim_allocator(1),
                    space_dim_allocator(1), space_dim_allocator(1) ))
D = Function(name='D', grid=gridD, space_order=2,
                dtype=np.float64)
print("D:", D.shape)
print("D data:", D.data_with_halo.shape)

"""
B: (2, 4, 3, 3)
B data: (2, 4, 3, 3)
"""

# print("dda1", default_dim_allocator(1))
print("space:", space_dim_allocator(1))
# print("space+:", space_dim_allocator(1)+ default_dim_allocator(1) )
#print ("dda +", default_dim_allocator(2)+default_dim_allocator(2)+ space_dim_allocator(1))
print("dda4 ", default_dim_allocator(4))