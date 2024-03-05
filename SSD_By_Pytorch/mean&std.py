"""
    求图像的mean、std
    正确的求解方法
"""
from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法
from torchvision.datasets import ImageFolder#用于导入图片数据集
import tqdm

means = [0,0,0]
std = [0,0,0]#初始化均值和方差
transform=ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
dataset_1=ImageFolder("../data_mask/MaskDataset/VOC/test/test/",transform=transform)#导入数据集的图片，并且转化为张量
dataset_2=ImageFolder("../data_mask/MaskDataset/VOC/train/train/",transform=transform)
num_imgs=len(dataset_1)+len(dataset_2)#获取数据集的图片数量
print(num_imgs)
for img,a in tqdm.tqdm(dataset_1,total=len(dataset_1),ncols = 100):#遍历数据集的张量和标签
    for i in range(3):#遍历图片的RGB三通道
        # 计算每一个通道的均值和标准差
        means[i] += img[i, :, :].mean()
        std[i] += img[i, :, :].std()

for img,a in tqdm.tqdm(dataset_2,total=len(dataset_2),ncols = 100):#遍历数据集的张量和标签
    for i in range(3):#遍历图片的RGB三通道
        # 计算每一个通道的均值和标准差
        means[i] += img[i, :, :].mean()
        std[i] += img[i, :, :].std()
mean=np.array(means)/num_imgs
std=np.array(std)/num_imgs#要使数据集归一化，均值和方差需除以总图片数量
print(f"mean: {mean}")#打印出结果
print(f"std: {std}")