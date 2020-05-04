from PIL import Image
from src.cfg import *
import os
import numpy as np

def center_crop(img,size=3000):
    w, h = img.size
    x_crop = (w-size)//2
    y_crop = (h-size)//2
    img = np.array(img)
    img = img[y_crop:h-y_crop,x_crop:w-x_crop,:]
    img = Image.fromarray(img)
    return img

# 原图裁剪
for f in os.listdir(IMAGE_ROOT):
    imgpath = os.path.join(IMAGE_ROOT,f)
    img = Image.open(imgpath).convert('RGB')

    assert isinstance(img , Image.Image)
    img = center_crop(img,2944)

    img.save(os.path.join(IMAGE_ROOT2,f))

# 标签裁剪
for f in os.listdir(LABEL_ROOT):
    imgpath = os.path.join(LABEL_ROOT,f)
    img = Image.open(imgpath).convert('RGB')

    assert isinstance(img , Image.Image)
    img = center_crop(img,2944)
    img.save(os.path.join(LABEL_ROOT2,f))

# 统计标签大小 #最小大概是 100左右
total_sizes = []
fns = []
with open("../dataset/blobsize.txt",'w') as fb:
    for f in os.listdir(LABEL_ROOT2):
        print(f)
        imgpath = os.path.join(LABEL_ROOT2,f)
        img = Image.open(imgpath).convert('L')
        img = np.array(img)
        totalsize = np.nonzero(img>0)

        blobsize = len(totalsize[0])
        print(blobsize)
        if blobsize>0:
            fns.append(f)
            total_sizes.append(blobsize)

    for b in total_sizes:
        if b>0:
            fb.write(f" {b}")
    print(sorted(total_sizes))

# 整理成文件名
with open("../dataset/datafile.txt",'w',encoding='utf-8') as data_file:
    for f in os.listdir(IMAGE_ROOT2):
        data_file.write(f)
        data_file.write(',')
        f= f.split('.')[0]+'.png'
        data_file.write(f)
        data_file.write('\n')
