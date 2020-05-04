#
# 没有裁剪前的路径 (太大了)
IMAGE_ROOT = 'F:/model/images'
LABEL_ROOT = 'F:/model/class_Feature'


'''
数据预处理执行(pre_dataset 执行)：
    原图裁剪
    标签裁剪
    整理成文件名 （dataset 图片路径组成的文件）
'''


#裁剪后存放的路径。大小： 2944*2944
IMAGE_ROOT2 = 'F:/model/_images'
LABEL_ROOT2 = 'F:/model/_class_Feature'

SAVE_TEST_IMAGE ='F:/model/images_test'

IMAGE_SIZE = 2944
RESCALE_SIZE = 768 #reisze 尺寸

TRAIN_BATCH = 2 #训练batch size
TEST_BATCH = 1 #测试batch size
# VAL_BATCH = 2