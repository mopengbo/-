import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.cuda.amp import autocast

"""
类别预测层 边界框预测层
使用保持长宽的卷积层代替全连接层 #输出与输入一一对应 共有(类别+1)*每个元素arche数个通道
"""
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1) #padding 四周填充
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 3, 512, 512)), cls_predictor(3, 3, 2))
Y2 = forward(torch.zeros((2, 6, 256, 256)), cls_predictor(6, 3, 2))
print(Y1.shape, Y2.shape)


"""
不同尺度的连接

除批次外，其他维扁平化为一维  flatten实现
其次将数据按===形式拼接  cat实现
"""
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

print(concat_preds([Y1, Y2]).shape)



"""
高宽减半块

卷积——>激活——>卷积——>激活——>池化
扩大感受野
"""
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))             #进入激活层之前先归一化
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

#print(forward(torch.zeros((2, 3, 512, 512)), down_sample_blk(3, 6)).shape)



"""
基本网络块
3个减半块相连
512*512 降为 64*64
"""
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

#print(forward(torch.zeros((2, 3, 512, 512)), base_net()).shape)



"""
完整模型

共5层：基本块 + 减半块*3 + 最大池化层
"""
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

#前向传播
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)                                                   #特征图输出
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)    #锚框
    cls_preds = cls_predictor(Y)                                 #预测类别
    bbox_preds = bbox_predictor(Y)                               #预测偏移量
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1





"""
定义SSD网络
"""
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))
        self.dropout = nn.Dropout(p=0.5)

    @autocast()
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)

        anchors = self.dropout(anchors)
        cls_preds = self.dropout(cls_preds)
        bbox_preds = self.dropout(bbox_preds)
        return anchors, cls_preds, bbox_preds



