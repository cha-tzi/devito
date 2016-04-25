from generator import Generator
import numpy as np
import cgen
from function_descriptor import FunctionDescriptor
from propagator import Propagator


class Test_Numpy_Array_Transfer(object):

    def test_2d(self):
        data = np.arange(6, dtype=np.float64).reshape((3, 2))
        kernel = cgen.Assign("output_grid[i2][i1]", "input_grid[i2][i1] + 3")
        p = Propagator(3, (2,))
        loop = p.prepare_loop(kernel)
        fd = FunctionDescriptor("process", loop)
        fd.add_matrix_param("input_grid", len(data.shape), data.dtype)
        fd.add_matrix_param("output_grid", len(data.shape), data.dtype)
        g = Generator([fd])
        f = g.get_wrapped_functions()[0]
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[2][1] == 8)

    def test_3d(self):
        kernel = cgen.Assign("output_grid[i3][i2][i1]",
                             "input_grid[i3][i2][i1] + 3")
        data = np.arange(24, dtype=np.float64).reshape((4, 3, 2))
        p = Propagator(4, (3, 2))
        loop = p.prepare_loop(kernel)
        fd = FunctionDescriptor("process", loop)
        fd.add_matrix_param("input_grid", len(data.shape), data.dtype)
        fd.add_matrix_param("output_grid", len(data.shape), data.dtype)
        g = Generator([fd])
        f = g.get_wrapped_functions()[0]
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[3][2][1] == 26)

    def test_4d(self):
        kernel = cgen.Assign("output_grid[i4][i3][i2][i1]",
                             "input_grid[i4][i3][i2][i1] + 3")
        data = np.arange(120, dtype=np.float64).reshape((5, 4, 3, 2))
        p = Propagator(5, (4, 3, 2))
        loop = p.prepare_loop(kernel)
        fd = FunctionDescriptor("process", loop)
        fd.add_matrix_param("input_grid", len(data.shape), data.dtype)
        fd.add_matrix_param("output_grid", len(data.shape), data.dtype)
        g = Generator([fd])
        f = g.get_wrapped_functions()[0]
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[4][3][2][1] == 122)
