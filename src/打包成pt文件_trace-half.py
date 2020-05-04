from src.Nets import *


import torch
import torch.nn as nn
from torch.jit import ScriptModule,script_method,trace

class ConvolutionalLayer2(nn.Module):
    def __init__(self):
        super(ConvolutionalLayer2, self).__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)

    # @torch.jit.script_method
    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':

    PATH = './logs/tb_unet/version_1/checkpoints/epoch=19.ckpt'
    premodel = LitModel.load_from_checkpoint(checkpoint_path=PATH, netmodel=AttU_Net())

    # model = premodel.netmodel
    premodel.eval()
    model = premodel.half()
    data = torch.rand(1, 3, 768, 768).half()
    m = torch.jit.trace(model.eval().cuda(), (data.cuda(),))
    torch.jit.save(m, 'attunet_half.pt')
    exit()

    # model = torch.jit.load('attunet_half.pt')
    # data = torch.rand(1, 3, 768, 768).half().cuda()
    #
    # print(model(data))