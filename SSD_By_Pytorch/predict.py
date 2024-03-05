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
from torchvision.transforms import Resize
from model import TinySSD
from torchvision import transforms
import  PIL
'''
导入图片
'''
resize = transforms.Resize([512, 512],interpolation=PIL.Image.BICUBIC)

X = torchvision.io.read_image('./imgs/predict.jpg').unsqueeze(0).float()
X = resize(X)
img = X.squeeze(0).permute(1, 2, 0).long()  #squeeze降维 unsqueeze升维 permute转换维度




device, net = d2l.try_gpu(), TinySSD(num_classes=2) #模型实例化
state = torch.load('./models/SSD_state_220.pt',map_location='cpu')#205 220
net.load_state_dict(state['model']) #加载模型

def predict(X):
    net.eval()
    X = X/255
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)  #使分类结果符合概率
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors) #（批量大小，锚框的数量，6）类别 置信度 坐标*4
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1] #非极大值抑制
    return output[0, idx]   #返回




def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.numpy())
    for row in output:
        cf = float(row[0])
        score = float(row[1])
        print(cf,score)

        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6].detach() * np.array((w, h, w, h))]
        score = '%.2f' % score
        if int(cf)==0:
            text = str(score) + "Mask"
            d2l.show_bboxes(fig.axes, bbox, text, 'g')
        else:
            text = str(score) + "No Mask"
            d2l.show_bboxes(fig.axes, bbox, text, 'r')
    fig = plt.gcf()
    fig.savefig('./result/result.png')
    plt.show()
output = predict(X) #锚数[类别 置信度 坐标*4]
for i in range(len(output[:])):
    print("第"+str(i+1)+"个目标")
    print(output[i])
display(img,output,0.03)
'''
cap = cv.VideoCapture(0)  # 摄像头检测
if (cap.isOpened()):  # 视频打开成功
    while (True):
        cap = cv.VideoCapture(0)  # 摄像头检测
        ret, frame = cap.read()  # 读取一帧
        fram = frame
        frame = cv.resize(frame, dsize=(512, 512), fx=2, fy=2)
        b, g, r = cv.split(frame)
        frame = cv.merge((r, g, b))
        img = torch.as_tensor(frame).permute(2, 0, 1).unsqueeze(0).float()

        result = predict(img / 255.0)
        for row in result:
            cf = float(row[0])
            score = float(row[1])
            # print(cf,score)

            if score < 0.8:
                continue
            h, w = 512,512
            bbox = [row[2:6].detach() * np.array((w, h, w, h))]
            x1 = int(bbox[0][0])
            y1 = int(bbox[0][1])
            x2 = int(bbox[0][2])
            y2 = int(bbox[0][3])
            score = '%.2f' % score
            cv.rectangle(fram, (x1, y1), (x2, y2), (255, 200, 0), 2)
            if int(cf) == 1:
                cv.putText(fram, 'No Mask', (x1, y1), 0, 0.8, (0, 0, 255), 2)
            else:
                cv.putText(fram, 'Mask', (x1, y1), 0, 0.8, (0, 255, 0), 2)
        cv.imshow('mask_detection', fram)
        if cv.waitKey(1) & 0xFF == 27:  # 按下Esc键退出
            break
cap.release()
cv.destroyAllWindows()

'''
