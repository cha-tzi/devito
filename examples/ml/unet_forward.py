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
batch_size = 2
conv1 = ml.ConvPad(kernel_size=(16, 3, 3),
                input_size=(batch_size, 3, 128, 128),
                padding=(1, 1),
                activation=relu,
                generate_code=False)

conv2 = ml.Conv(kernel_size=(16, 3, 3),
                input_size=(batch_size, 16, 130, 130),
                #padding=(1, 1),
                activation=relu,
                generate_code=False)

pool1 = ml.SubsamplingPad(kernel_size=(2, 2),
                       input_size=(batch_size, 16, 128, 128),
                       function=maximum,
                       stride=(2, 2),
                       strict_stride_check=False,
                       generate_code=False)

conv3 =  ml.ConvPad(kernel_size=(32, 3, 3),
                input_size=(batch_size, 16, 64, 64),
                padding=(1, 1),
                activation=relu,
                generate_code=False)

conv4 =  ml.Conv(kernel_size=(32, 3, 3),
                input_size=(batch_size, 32, 66, 66),
                #padding=(1, 1),
                activation=relu,
                generate_code=False)

pool2 = ml.SubsamplingPad(kernel_size=(2, 2),
                       input_size=(batch_size, 32, 64, 64),
                       function=maximum,
                       stride=(2, 2),
                       strict_stride_check=False,
                       generate_code=False)

# for the second convolution of the double conv
# we remove the padding since it was applied by the above layer
# and we reflect that on the input size
"""
eqs = []
eqs += conv1.equations(input_function=None)
eqs += conv2.equations(input_function=conv1.result)
eqs += pool1.equations(input_function=conv2.result)
"""
layers = [conv1, conv2, pool1, conv3, conv4, pool2]  
eqs = []
input_function = None
    
for layer in layers:
    eqs += layer.equations(input_function=input_function)
    input_function = layer.result

OP = Operator(eqs)

conv1.kernel.data[:] = torch.randn((16, 3, 3, 3))
conv1.bias.data[:] = torch.randn((16,))

conv2.kernel.data[:] = torch.randn((16, 16, 3, 3))
conv2.bias.data[:] = torch.randn((16,))

conv3.kernel.data[:] = torch.randn((32, 16, 3, 3))
conv3.bias.data[:] = torch.randn((32,))

conv4.kernel.data[:] = torch.randn((32, 32, 3, 3))
conv4.bias.data[:] = torch.randn((32,))

INPUT_DATA = torch.rand((batch_size, 3, 128, 128))
conv1.input.data[..., 1:-1, 1:-1] = INPUT_DATA

print("conv1.input.data.data", conv1.input.data.shape)
print("conv1.result.data", conv1.result.shape)
print("conv2.input.data", conv2.input.data.shape)
print("conv2.result.data", conv2.result.shape)
print("pool1 input data", pool1.input.data.shape)
print("pool1 result data", pool1.result.shape)
print("conv3.input.data", conv3.input.data.shape)
print("conv3.result.data", conv3.result.shape)
print("conv4.input.data", conv4.input.data.shape)
print("conv4.result.data", conv4.result.shape)
print("pool2 input data", pool2.input.data.shape)
print("pool2 input data", pool2.result.shape)

OP.apply()

print("Final result ", layers[-1].result.data)
print("Final result shape", layers[-1].result.data.shape)


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
