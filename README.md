基于 attention UNet 的缺陷检测

使用的框架：
pytorch 
torchvision
pytorch的封装pytorch lightning
apex 混合精度训练框架


检测对象：纸碗
缺陷：类似于脏污

数据集组成（不进行公开）：
共 422 个：训练集：372 测试集：50

效果：example.png

结果：
    {'acc': 0.9998152852058411,
    'mIOU': 0.6120708584785461}