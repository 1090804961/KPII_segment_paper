import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.Nets import *
from torchvision import transforms
import torch.nn.functional as F
from pytorch_lightning import loggers
import time
import torch
from torch import jit
# def totaltime(fn):
#     st = time.time()
#     fn()
#     endt = time.time()
#     return endt - st

""".test() is not stable yet on TPUs. We’re working on getting around the multiprocessing challenges."""
PATH = './logs/tb_unet/version_1/checkpoints/epoch=19.ckpt'
premodel = LitModel.load_from_checkpoint(checkpoint_path=PATH,netmodel=AttU_Net())

tb_logger = loggers.TensorBoardLogger('logs/',name='tb_unet')

trainer = Trainer(gpus=[0])

premodel.eval()
premodel.freeze()
premodel= premodel.half().cuda()

st = time.time()
trainer.test(premodel)
endt = time.time()
t = endt - st
print(f"总时间：{t}，平均时间：{t/len(premodel.test_dataloader())}")
exit()


#-----------------------------------以下是使用自定义for循环，比lightning框架快-----------------------------------------
premodel.prepare_data()
dloader = premodel.test_dataloader()
totaltime = time.time()
#jit：# model = torch.jit.load("attunet_half.pt",map_location=torch.device('cuda'))

count = 0
for i,(d,l) in enumerate(dloader):
    starttime = time.time()

    #jit：# out = model.forward(d.cuda().type(torch.float16)) #gpu占用异常，主要在c++端使用

    out = premodel(d.cuda().type(torch.float16))
    #
    out = torch.sigmoid(out) #激活到0-1（概率）


    ximg = d * 0.5 + 0.5 # 反 标准化

    lblimg = l.repeat(1, 3, 1, 1).float()#单通道的标签图扩展为三通道，和输入图片一起拼接输出用的
    # outimg = (out > 0.5).type_as(ximg).repeat(1, 3, 1, 1) #单通道的结果图扩展为三通道，本来和输入图片一起输出用的

    #缺陷处涂色:[N,C,H,W]
    ximg[:,0,out[0,0] > 0.5] = 255  # 控制第一个通道的幅值 # 0.043左右

    img = torch.cat((ximg, lblimg), dim=3) #0.007左右
    # # 输出保存本地
    outimage = tfs_F.to_pil_image(img[0].detach().cpu(), mode='RGB')
    outimage.save(f'{SAVE_TEST_IMAGE}/{i}.jpg')
    #本次的时间
    endtime = time.time()
    print(f"当前 {i+1}/ {len(dloader)}，usetime ：{endtime - starttime}")
    count+=1
    torch.cuda.empty_cache()
print(f'平均时间：{(time.time() - totaltime)/count} 秒')
print(f'总计时间：{time.time() - totaltime} 秒')