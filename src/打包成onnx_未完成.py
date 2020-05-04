from src.Nets_Script import *


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.jit import ScriptModule,script_method,trace


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, input):
        return input
        # return F.interpolate(input, scale_factor=2., mode="bilinear", align_corners=True)

if __name__ == '__main__':

    example = (torch.rand(1,3,20,20),)

    torch_out = torch.onnx.export(MyModule(),  # model being run
                                  example,  # model input (or a tuple for multiple inputs)
                                  "conv.onnx",
                                  opset_version=11,# this set ，support interpolate
                                  verbose=False,  # store the trained parameter weights inside the model file
                                  training=False,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output'])




