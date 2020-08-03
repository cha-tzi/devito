# Test of TransConv againgst Pytorch's ConvTranspose2d
# =======SUCCESS=================
import torch
import devito.ml as ml
from devito import Operator
import sympy
import numpy as np

input_data = torch.rand((2,4,5,5))
w1 = torch.ones((4, 2, 2, 2))
for i in range(4):
    w1[i] = w1[i]+i+1

torch_trans_noset = torch.nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2)
#  Number of channels in the input image, Number of channels produced by the convolution
#torch_trans.weight =torch.nn.Parameter(torch.ones(3, 1, 2, 2)+2.42323)

torch_trans_noset.weight = torch.nn.Parameter(w1)
weight_dev = torch_trans_noset.weight.transpose(0,1).detach().numpy()

torch_trans_noset.bias = torch.nn.Parameter(torch.zeros((2,)))
bias_dev = torch_trans_noset.bias.detach().numpy()

torch_res_trans = torch_trans_noset(input_data)

layer1 = ml.TransConv(kernel_size=(2, 2, 2),
                       input_size=(2,4,5,5),
                       stride = (2,2))
current_data = layer1.execute(input_data,
        bias_dev, weight_dev ) #w1.transpose(0,1)
print("Same result?", np.allclose(torch_res_trans.detach().numpy(), current_data))
print("error:", np.sum(abs(abs(torch_res_trans.detach().numpy()) - abs(current_data))))
print("torch shape:", torch_res_trans.shape)
print("Devito shape:", current_data.shape )