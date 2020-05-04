from src.Nets_Script import *


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.jit import ScriptModule,script_method,trace,script

class ConvolutionalLayer2(torch.jit.ScriptModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer2, self).__init__()

        self.sub_module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    @torch.jit.script_method
    def forward(self, x):
        return self.sub_module(x)

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2.0, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.up =nn.MaxPool2d(2)

    def forward(self, X):
        # x= F.interpolate(X, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return X

    # nn.functional.

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(3,3,3,2,1,1,bias=False)
                                ,nn.BatchNorm2d(3)
                                ,nn.LeakyReLU(0.1)
                                )

    def forward(self, input):
        return F.interpolate(input,scale_factor=2,mode='nearest')

class Up_Cat(nn.Module):
    def __init__(self):
        super(Up_Cat, self).__init__()

    def forward(self,input):
        x = F.interpolate(input,scale_factor=2,mode='nearest')#bilinear
        return x
if __name__ == '__main__':
    # data = torch.FloatTensor(1, 1, 256, 256)
    # model = Up_Cat()
    # traced_script_module = torch.jit.trace(model, data)
    # traced_script_module.save('up.pt')
    # mm = torch.jit.load(r'C:\Users\ZY\Desktop\new_unet\KPII_segment_paper\model.pt')
    # print(mm.graph)


    data = torch.rand(1, 3, 768, 768)

    # my_module = MyModule().eval()

    traced_script_module = AttU_Net()

    traced_script_module = torch.jit.trace(traced_script_module,(data,))

    traced_script_module.save("conv.pt")
    print(traced_script_module.graph)
    # print(traced_script_module(data).shape)
    exit()



