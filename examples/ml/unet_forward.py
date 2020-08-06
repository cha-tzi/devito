# unet forward
import devito.ml as ml
from devito import Operator
import sympy
import numpy as np
import torch
def relu(x):
    return sympy.Max(0, x)
def maximum(lst):
    return sympy.Max(*lst)

conv1 = ml.Conv(kernel_size=(16, 3, 3),
                input_size=(2, 3, 128, 128),
                padding=(1, 1),
                activation=relu,
                generate_code=False)

conv2 = ml.Conv(kernel_size=(16, 3, 3),
                input_size=(2, 16, 128, 128),
                #padding=(1, 1),
                activation=relu,
                generate_code=False)
eqs = []
eqs += conv1.equations(input_function=None)
eqs += conv2.equations(input_function=conv1.result)
#data_with_halo

OP = Operator(eqs)

conv1.kernel.data[:] = torch.randn((16, 3, 3, 3))
conv1.bias.data[:] = torch.randn((16,))

conv2.kernel.data[:] = torch.randn((16, 16, 3, 3))
conv2.bias.data[:] = torch.randn((16,))

INPUT_DATA = torch.rand((2, 3, 128, 128))
conv1.input.data[..., 1:-1, 1:-1] = INPUT_DATA
# print("conv1.input.data", conv1.input.data.shape)
# print("conv2.input.data", conv2.input.data.shape)
# print("conv1.result", conv1.result.shape)
OP.apply()
print("conv2 shape:", conv2.result.data.shape)

print("conv1.result with pad attept")
#print(conv1.result.data_with_halo)


"""
2 operators, works fine
conv1 = ml.Conv(kernel_size=(16, 3, 3),
                input_size=(2, 3, 128, 128),
                padding=(1, 1),
                activation=relu,
                generate_code=False)
conv1.kernel.data[:] = torch.randn((16, 3, 3, 3))
conv1.bias.data[:] = torch.randn((16,))
INPUT_DATA = torch.rand((2, 3, 128, 128))
conv1.input.data[..., 1:-1, 1:-1] = INPUT_DATA
op = Operator(conv1.equations())
result = op.apply()
print("Conv1 result shape:", conv1.result.data.shape)

conv2 = ml.Conv(kernel_size=(16, 3, 3),
                input_size=(2, 16, 128, 128),
                padding=(1, 1),
                activation=relu,
                generate_code=False)
conv2.kernel.data[:] = torch.randn((16, 16, 3, 3))
conv2.bias.data[:] = torch.randn((16,))



conv2.input.data[...,1:-1,1:-1] = conv1.result.data
op2 = Operator(conv2.equations())
result2 = op2.apply()
print("conv2 shape:", conv2.result.data.shape)
"""
