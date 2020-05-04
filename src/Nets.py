import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split,DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import functional as tfs_F

from src.cfg import *
from src.mDataset import *

import collections
# -----------------------------------------
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        return self.sub_module(x)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        return self.sub_module(x)

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()

        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x = nn.functional.interpolate(x,scale_factor=2.,mode='bilinear',align_corners=False)
        return self.sub_module(x)

channels = [16,32,64,128,256,512,1024]
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels=img_ch, out_channels=channels[1])
        self.Conv2 = conv_block(in_channels=channels[1], out_channels=channels[2])
        self.Conv3 = conv_block(in_channels=channels[2], out_channels=channels[3])
        self.Conv4 = conv_block(in_channels=channels[3], out_channels=channels[4])
        self.Conv5 = conv_block(in_channels=channels[4], out_channels=channels[5])

        self.Up5 = up_conv(in_channels=channels[5], out_channels=channels[4])
        self.Att5 = Attention_block(F_g=channels[4], F_l=channels[4], F_int=channels[3])
        self.Up_conv5 = conv_block(in_channels=channels[5], out_channels=channels[4])

        self.Up4 = up_conv(in_channels=channels[4], out_channels=channels[3])
        self.Att4 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=channels[2])
        self.Up_conv4 = conv_block(in_channels=channels[4], out_channels=channels[3])

        self.Up3 = up_conv(in_channels=channels[3], out_channels=channels[2])
        self.Att3 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[1])
        self.Up_conv3 = conv_block(in_channels=channels[3], out_channels=channels[2])

        self.Up2 = up_conv(in_channels=channels[2], out_channels=channels[1])
        self.Att2 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[0])
        self.Up_conv2 = conv_block(in_channels=channels[2], out_channels=channels[1])

        self.Conv_1x1 = nn.Conv2d(channels[1], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), #输出通道 C = 1
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g) #上一层注意力输入
        x1 = self.W_x(x)# 当前层注意力输入
        psi = self.relu(g1 + x1) #合并上一层和当前层的注意力计算
        psi = self.psi(psi) #(0-1) 这里进行了注意力计算

        return x * psi

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confusionMatrix = torch.zeros((self.numClass,) * 2).to(device)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(dim=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = torch.mean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)
        union = torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) - torch.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = torch.mean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label.long(), minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = torch.sum(self.confusionMatrix, dim=1) / torch.sum(self.confusionMatrix)
        iu = torch.diag(self.confusionMatrix) / (
                torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) -
                torch.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))

class LitModel(pl.LightningModule):
    #定义网络
    def __init__(self,netmodel):
        super(LitModel, self).__init__()
        self.netmodel = netmodel
        self.seg_Metric = SegmentationMetric(2)

    def forward(self, x):
        x = self.netmodel(x)
        return x

    #定义优化器
    def configure_optimizers(self):
        return Adam(self.parameters(),lr = 5e-4)

    def prepare_data(self):
        mdataset = mDataset(IMAGE_ROOT2,LABEL_ROOT2)
        torch.manual_seed(1) #固定随机种子
        # self.traindata,self.testdata,self.valdata = random_split(mdataset,[350,36,36])
        self.traindata, self.testdata= random_split(mdataset, [372, 50])
        torch.seed()#将随机种子设置为随机

    def training_step(self, batch,batch_idx):
        x,y = batch

        #显示 模型的图
        '''在半精度训练时，这里会报错'''
        # if batch_idx ==1:
        #     self.logger.experiment.add_graph(self.netmodel.half(),x)

        out = self(x) #网络输出结果

        loss = nn.functional.binary_cross_entropy_with_logits(out,y.float())

        # if batch_idx%20:
        #     self.logger.experiment.add_scalar('loss', loss,batch_idx)

        self.logger.experiment.add_scalar('loss', loss, batch_idx)

        #使用自带的进行log 会产生多余的文件夹

        # logger_logs = {'training_loss': loss} #配合下面的output，好像会自动log
        output = {
            'loss': loss,  # required
            # 'log': logger_logs
        }
        return output

    def test_step(self,batch,batch_idx):
        x, y = batch
        x, y = x.half(), y.half()
        out = self(x)
        out = torch.sigmoid(out)
        self.seg_Metric.addBatch(out, y)
        acc = self.seg_Metric.pixelAccuracy()
        mIoU = self.seg_Metric.meanIntersectionOverUnion()

        ximg = x.float()*0.5+0.5
        lblimg = y.repeat(1,3,1,1).float()
        # outimg = (out>0.5).type_as(ximg).repeat(1,3,1,1) #输出的3通道预测图片

        #标记输出图片颜色
        ximg[:, 1][out[:, 0] > 0.5] = 255  # 控制第一个通道的幅值。 原图缺陷区域染色
        ximg[:, 2][out[:, 0] > 0.5] = 255  # 控制第一个通道的幅值。 原图缺陷区域染色

        img = torch.cat((ximg,lblimg),dim=3)

        #图片输出到tensorboard
        # self.logger.experiment.add_images(f"images{batch_idx}", img, batch_idx)

        #输出保存本地
        outimage = tfs_F.to_pil_image(img[0].detach().cpu(), mode='RGB')
        outimage.save(f'{SAVE_TEST_IMAGE}/{batch_idx}.jpg')

        output = collections.OrderedDict({
            'acc': acc.clone().detach(),
            'mIoU': mIoU.clone().detach(),  # everything must be a tensor
            # 'img':img.clone().detach() #这边输出图片，导致内存越来越大
        })
        del outimage
        del img
        del x,y
        return output

    def test_epoch_end(self, outputs):
        test_acc_mean = 0
        test_iou_mean = 0

        count = 1
        for output in outputs:
            test_acc_mean += output['acc']
            test_iou_mean += output['mIoU']
            # imgs.append(outputs['img'])
            # if count%3 ==0 :
            # self.logger.experiment.add_images(f"images{count}", output['img'], count)

            count += 1

        test_acc_mean /= len(outputs)
        test_iou_mean /= len(outputs)

        # self.logger.experiment.add_scalars('result', {'acc':test_acc_mean.item(),'result':test_iou_mean.item()})

        tqdm_dict = {"test_iou_mean": test_iou_mean} #这一句还不知道有什么用
        results = {
            'progress_bar': tqdm_dict,
            'log': {'acc':test_acc_mean.item(),'result':test_iou_mean.item()} #这里的log只能输出到控制台
        }

        return results


    def train_dataloader(self):
        return DataLoader(self.traindata,TRAIN_BATCH,True,drop_last=True)
    def test_dataloader(self):
        dl = DataLoader(self.testdata, TEST_BATCH, False, drop_last=True)
        return dl
    # def val_dataloader(self):
    #     return DataLoader(self.val, TRAIN_BATCH, True, drop_last=True)


