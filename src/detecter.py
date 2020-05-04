import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.Nets import *
from torchvision import transforms
import torch.nn.functional as F
from pytorch_lightning import loggers
import time
import torch
from torch import jit
def totaltime(fn):
    st = time.time()
    fn()
    endt = time.time()
    return endt - st

""".test() is not stable yet on TPUs. We’re working on getting around the multiprocessing challenges."""
PATH = './logs/tb_unet/version_1/checkpoints/epoch=19.ckpt'
premodel = LitModel.load_from_checkpoint(checkpoint_path=PATH,netmodel=AttU_Net())

tb_logger = loggers.TensorBoardLogger('logs/',name='tb_unet')
# trainer = Trainer(gpus=1,logger=tb_logger)

trainer = Trainer(gpus=[0])

premodel.eval()
premodel.freeze()
premodel= premodel.half().cuda()
# st = time.time()
# trainer.test(premodel)
# endt = time.time()
# t = endt - st
# print(f"总时间：{t}，平均时间：{t/72}")

premodel.prepare_data()
dloader = premodel.test_dataloader()

totaltime = time.time()

# model = torch.jit.load("attunet_half.pt",map_location=torch.device('cuda'))

count = 0
for i,(d,l) in enumerate(dloader):
    starttime = time.time()
    # if i==10:break

    # out = model.forward(d.cuda().type(torch.float16))

    out = premodel(d.cuda().type(torch.float16))
    #
    out = torch.sigmoid(out)
    #
    #
    # ximg = d * 0.5 + 0.5
    #
    # lblimg = l.repeat(1, 3, 1, 1).float()
    # outimg = (out > 0.5).type_as(ximg).repeat(1, 3, 1, 1)
    #
    # ximg[:,0,out[0,0] > 0.5] = 255  # 控制第一个通道的幅值 # 0.043左右
    #
    # img = torch.cat((ximg, lblimg), dim=3) #0.007左右
    # # # 输出保存本地
    # outimage = tfs_F.to_pil_image(img[0].detach().cpu(), mode='RGB')
    # outimage.save(f'{SAVE_TEST_IMAGE}/{i}.jpg')
    endtime = time.time()
    print(f"当前 {i+1}/ {len(dloader)}，usetime ：{endtime - starttime}")
    count+=1
    # exit()
    torch.cuda.empty_cache()
# totaltime = (time.time() - totaltime)/len(dloader)
print(f'平均时间：{(time.time() - totaltime)/count} 秒')
print(f'总计时间：{time.time() - totaltime} 秒')