# conv pad test
# padding.py
# unet forward
import sympy
import torch
import numpy as np
import devito.ml as ml
from devito import Operator
def relu(x):
    return sympy.Max(0, x)
def maximum(lst):
    return sympy.Max(*lst)
batch_size, channels_in, height, width = (2, 3, 128, 128)
channels_out, height_kernel, width_kernel = (16, 3, 3)
stride = 1
padding = 1
INPUT_DATA = torch.rand((batch_size, channels_in, height, width))
w1 = torch.ones((channels_out, channels_in, height_kernel, width_kernel))
for i in range(3):
    w1[i] = w1[i]+i+1

torch_conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size=width_kernel, stride=stride, padding=padding)
#  Number of channels in the input image, Number of channels produced by the convolution

torch_conv.weight = torch.nn.Parameter(w1)
#weight_dev = torch_conv.weight.transpose(0,1).detach().numpy()
weight_dev = torch_conv.weight.detach().numpy()

torch_conv.bias = torch.nn.Parameter(torch.zeros((channels_out,)))
bias_dev = torch_conv.bias.detach().numpy()

torch_result = torch_conv(INPUT_DATA)

conv1 = ml.Conv(kernel_size=(channels_out, height_kernel, width_kernel),
                input_size=(batch_size, channels_in, height, width),
                padding=(padding, padding),
                activation=relu,
                generate_code=False)

#devito_result = conv1.execute(INPUT_DATA, bias_dev, weight_dev)
# print("Same result?", np.allclose(torch_result.detach().numpy(), devito_result))
# print("error:", np.sum(abs(abs(torch_result.detach().numpy()) - abs(devito_result))))
# print("torch shape:", torch_result.shape)
# print("Devito shape:", devito_result.shape)


conv2 = ml.Conv(kernel_size=(channels_out, height_kernel, width_kernel),
                input_size=(batch_size, channels_out, height, width),
                padding=(padding, padding),
                activation=relu,
                generate_code=False)
eqs = []
eqs += conv1.equations(input_function=None)
eqs += conv2.equations(input_function=conv1.result)

OP = Operator(eqs)

conv1.kernel.data[:] = weight_dev
conv1.bias.data[:] = bias_dev

conv2.kernel.data[:] = torch.randn((16, 16, 3, 3))
conv2.bias.data[:] = torch.randn((16,))

# INPUT_DATA = torch.rand((2, 3, 128, 128))
# conv1.input.data[..., 1:-1, 1:-1] = INPUT_DATA
conv1.input.data[:] = INPUT_DATA
print("conv1.input.data", conv1.input.data_with_halo.shape)
print("conv1.result", conv1.result.shape)

print("conv2.input.data", conv2.input.data.shape)
print("conv2.result", conv2.result.shape)
OP.apply()

print("conv2 shape:", conv2.result.data.shape)

print("torch result", torch_result.shape)

print("Same result?", np.allclose(torch_result.detach().numpy(), conv1.result.data))
print("error:", np.sum(abs(abs(torch_result.detach().numpy()) - abs(conv1.result.data))))
print("conv1.result.data", conv1.result.data[0][2])
print("torch result", torch_result[0][2])