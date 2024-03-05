from flask import Flask, render_template,request
from flask import send_file
from predict import *
from model import TinySSD
import os,base64
import jieba
import re
from d2l import torch as d2l
import numpy as np
import cv2
import torch

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
app = Flask(__name__)

'''
@app.route("/")
def index_1():
    return render_template('index.html')

@app.route("/login/",methods=['GET','POST'])
def login():
    name=request.form.get("name",type=str, default=None)
    print(name)
    return render_template('index_2.html',name=name)

@app.route("/learn/<name>")
def learn(name):
    yours = request.args.get("your")
    return "我的名字是 "+name+"\n"+"你的名字是 "+str(yours)

'''




@app.route("/")
def geter():
    return render_template("index.html")

@app.route("/main_use/",methods=['GET','POST'])
def main_use():
    #data=request.form.get("imgfile")

    load_img = request.files['imgfile'].read()
    load_img = np.frombuffer(load_img, dtype=np.uint8)
    load_img = cv2.imdecode(load_img, cv2.IMREAD_COLOR)
    load_img = cv.resize(load_img, dsize=(512,512), fx=2, fy=2)
    cv2.imwrite('./static/load_img.png',load_img)
    b, g, r = cv.split(load_img)
    load_img = cv.merge((r, g, b))
    img = torch.as_tensor(load_img).permute(2,0,1).unsqueeze(0).float()

    if request.form.get('type')=="ssd":
        print("ssd")
        result=predict(img/255.0)
        img = img.squeeze(0).permute(1, 2, 0).long()
        display(img, result, threshold=0.05)


    #return send_file('./result/result.png')
        import base64
        img_stream = ''
        with open('./result/result.png', 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        return render_template('index.html',
                        result=img_stream)



if __name__ == '__main__':
    app.run(debug=True)