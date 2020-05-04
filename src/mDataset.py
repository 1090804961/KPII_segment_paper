from torch.utils.data import Dataset,random_split
from torchvision import transforms as tfs
from PIL import Image
from src.cfg import *
import os


# 定义transforms的一些操作
data_transform = tfs.Compose([
		# # Resize后数据的大小为224 * 224
        # tfs.RandomResizedCrop(224),
        # tfs.RandomHorizontalFlip(),
        tfs.Resize((RESCALE_SIZE,RESCALE_SIZE)),
        tfs.ToTensor(),
        # 数据标准化，采用的图片标准化参数
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
label_transform = tfs.Compose([
		# # Resize后数据的大小为224 * 224
        # tfs.RandomResizedCrop(224),
        # tfs.RandomHorizontalFlip(),
        tfs.Resize((RESCALE_SIZE,RESCALE_SIZE)),
        tfs.ToTensor(),
    ])
class mDataset(Dataset):
    def __init__(self,imgroot,lblroot,filepath ='../dataset/datafile.txt'):
        super(mDataset, self).__init__()
        self.datalist = []
        self.labellist = []

        with open(filepath,'r',encoding='utf-8') as datafile:
            for data in datafile:
                data = data.strip()
                fn ,ln= data.strip().split(',')
                self.datalist.append(fn)
                self.labellist.append(ln)


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        imgname = os.path.join(IMAGE_ROOT2,self.datalist[index])
        lblname = os.path.join(LABEL_ROOT2,self.datalist[index].split('.')[0]+'.png')

        img = Image.open(imgname).convert('RGB')
        lbl = Image.open(lblname).convert('RGB')

        imgdata = data_transform(img)
        lbldata = label_transform(lbl)
        lbldata = lbldata[0:1]>0
        return imgdata,lbldata

if __name__ == '__main__':
    md = mDataset(IMAGE_ROOT2,LABEL_ROOT2)

    # print(len(md))
    #
    # imgdata,lbldata = md[0]
    # print(imgdata.shape,lbldata.shape)