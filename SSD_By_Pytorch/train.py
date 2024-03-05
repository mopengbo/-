import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
import cv2 as cv
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from model import TinySSD
from data_load import load_data
from data_load import *
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast


'''
# 图像的小批量的形状为（批量大小、通道数、高度、宽度）
# 标签的小批量的形状为（批量大小，m，5）
batch_size, edge_size = 16, 256
train_iter, _ = load_data(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

print(batch[0][0:2].shape)
imgs = (batch[0][0:2].permute(0, 2, 3, 1)) / 255
print('imgs形状'+str(imgs.shape))
i=0
for img in imgs:

    axes = d2l.show_images(img, 1, 1, scale=5)
    img2 = cv.resize(img.numpy(), None, fx=2, fy=2)
    b, g, r = cv.split(img2)
    img2 = cv.merge((r, g, b))
    #print((img2.shape))
    #print(batch[1][i][0].shape)
    cv.imshow("cv",img2)
    #cv.imwrite('./imgs/pic_'+str(i)+'.jpg', img2*255)
    d2l.show_bboxes(axes, [batch[1][i][0][1:5] * edge_size], colors=['w'])
    axes.imshow(img)
    plt.show()
    i=i+1
'''








"""
模型训练
"""
batch_size = 8
train_iter, _ = load_data(batch_size) #加载数据
device, net = d2l.try_gpu(), TinySSD(num_classes=2) #模型实例化

optim = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4) #parameter返回模型参数 #optim 优化器
torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.0001,
                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=3, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#定义损失函数
cls_loss = nn.CrossEntropyLoss(reduction='none') #分类问题 交叉熵损失
#cls_loss = FocalLoss()  #焦点损失
bbox_loss = torch.nn.SmoothL1Loss(reduction='none', beta=1.0)   #回归问题 L1范数

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """
    掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算
    将锚框类别和偏移量的损失相加，以获得模型的最终损失函数
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

#评价预测效果
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


"""
训练过程
"""
num_epochs, timer = 300, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
net.load_state_dict(torch.load('./models/SSD_state_220.pt')['model']) #加载模型
optim.load_state_dict(torch.load('./models/SSD_state_220.pt')['optimizer'])
for epoch in range(251,301):  #训练轮次

    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        optim.zero_grad()    #每轮梯度清零
        X, Y = features.to(device), target.to(device)

        with autocast():
            anchors, cls_preds, bbox_preds = net(X)  # 生成多尺度的锚框，为每个锚框预测类别和偏移量

            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)  # 为每个锚框标注类别和偏移量

            los = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,  # 根据类别和偏移量的预测和实际值计算损失函数
                            bbox_masks)
            los.mean().backward()  # 反向梯度传播
            optim.step()  # 更新参数

            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())

    if (epoch)%5==0:
        state = {'model': net.state_dict(), 'optimizer': optim.state_dict(), 'epoch': epoch}
        torch.save(state, './models/SSD_state_'+str(epoch)+'.pt')
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
plt.show()
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')


"""
保存模型
"""
#torch.save(net, './models/SSD_model.pt')
#torch.save(net.state_dict(), './models/SSD_state.pt')


"""
优化
optm学习率递减 wd
epoch 100
数据resize256
基本块不变
图像增强
/255
bantch 16
改善损失函数
增加dropout
半精度训练
"""










