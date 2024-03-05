from flask import Flask, render_template,request
import jieba
import re
import numpy as np
import os,base64
from io import BytesIO
from PIL import Image
from main import *
import cv2
from flask import Flask, jsonify, request, render_template
import os

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def geter():
    return render_template("demo.html")

@app.route("/main_use/",methods=['GET','POST'])
def main_use():
    if request.form.get('type')=="stream":
        return render_template('flask_stream.html')
    elif request.form.get('type')=="pic":
        return render_template('flask_pic.html')
    else:
        return render_template('flask_mp4.html')


@app.route("/stream_predict/",methods=['GET','POST'])
def stream_predict():
    image_data = request.form['image']
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    #img = image.load()
    img = np.array(image)#<class 'numpy.ndarray'>(480, 640, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TODO: 目标检测模型推理

    input_dict = {"data": [img]}
    result = mask_detector.face_detection(data=input_dict)
    count = len(result[0]['data'])
    if count < 1:
        # print('There is no face detected!')
        pass
    else:
        for i in range(0, count):
            # print(result[0]['data'][i])
            label = result[0]['data'][i].get('label')
            score = float(result[0]['data'][i].get('confidence'))
            x1 = int(result[0]['data'][i].get('left'))
            y1 = int(result[0]['data'][i].get('top'))
            x2 = int(result[0]['data'][i].get('right'))
            y2 = int(result[0]['data'][i].get('bottom'))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
            if label == 'NO MASK':
                cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 255, 0), 2)
    #cv2.imshow('mask_detection', img)
    img = cv2.resize(img, dsize=(640, 485), fx=1, fy=1)
    cv2.imwrite('./static/stream_result.jpg', img)
    with open('./static/stream_result.jpg', 'rb') as img_f:
        img_stream = img_f.read()
        print(type(img_stream))
        img_stream = base64.b64encode(img_stream).decode()
        print(type(img_stream))
    #return render_template('flask_web.html',img_stream)
    return img_stream

@app.route("/pic_predict/",methods=['GET','POST'])
def pic_predict():
    load_img = request.files['imgfile'].read()
    load_img = np.frombuffer(load_img, dtype=np.uint8)
    print(type(load_img))
    load_img = cv2.imdecode(load_img, cv2.IMREAD_COLOR)
    print(type(load_img))
    cv2.imwrite('./static/load_img.jpg',load_img)
    img = load_img
    input_dict = {"data": [img]}
    result = mask_detector.face_detection(data=input_dict)
    count = len(result[0]['data'])
    if count < 1:
        pass
    else:
        for i in range(0, count):
            # print(result[0]['data'][i])
            label = result[0]['data'][i].get('label')
            score = float(result[0]['data'][i].get('confidence'))
            x1 = int(result[0]['data'][i].get('left'))
            y1 = int(result[0]['data'][i].get('top'))
            x2 = int(result[0]['data'][i].get('right'))
            y2 = int(result[0]['data'][i].get('bottom'))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
            if label == 'NO MASK':
                cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 255, 0), 2)
    cv2.imwrite('./static/pic_result.jpg', img)
    return render_template('flask_pic.html')


'''
@app.route('/mp4_predict/', methods=['POST'])
def upload_chunk():
    
    chunk = request.files['chunk']
    start = int(request.form['start'])
    end = int(request.form['end'])
    total_size = int(request.form['totalSize'])
    filename = 'video.mp4'

    with open(filename, 'ab') as f:
        f.seek(start)
        bt = chunk.read()
        f.write(bt)
        bt = np.frombuffer(bt, dtype=np.uint8)
        print(bt)
        print(bt.shape)

    if end >= total_size:
        print('end')

    return 'OK'
'''
import base64
import numpy as np
@app.route('/mp4_predict/', methods=['POST'])
def mp4_predict():
    # 获取上传的视频文件
    video_file = request.files['video']
    video_file.save('video.mp4')
    #load_img = request.files['video'].read()
    #load_img = np.frombuffer(load_img, dtype=np.uint8)
    #frame = cv2.imdecode(load_img, cv2.IMREAD_COLOR)
    #print(type(frame))
    #print(str(frame))

    # 读取视频文件<class 'cv2.VideoCapture'>
    video = cv2.VideoCapture('video.mp4')
    print(video)
    # 获取视频帧率
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # 获取视频总帧数
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    # 遍历视频的每一帧
    for frame_index in range(total_frames):
        # 读取当前帧
        ret, frame = video.read()
        # 如果读取失败，跳过
        if not ret:
            continue

        # 对当前帧进行后续操作
        img = frame
        print(img.shape)
        input_dict = {"data": [img]}
        result = mask_detector.face_detection(data=input_dict)
        count = len(result[0]['data'])
        if count < 1:
            # print('There is no face detected!')
            pass
        else:
            for i in range(0, count):
                # print(result[0]['data'][i])
                label = result[0]['data'][i].get('label')
                score = float(result[0]['data'][i].get('confidence'))
                x1 = int(result[0]['data'][i].get('left'))
                y1 = int(result[0]['data'][i].get('top'))
                x2 = int(result[0]['data'][i].get('right'))
                y2 = int(result[0]['data'][i].get('bottom'))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                if label == 'NO MASK':
                    cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(img, label, (x1, y1), 0, 0.8, (0, 255, 0), 2)
        # cv2.imshow('mask_detection', img)
        img = cv2.resize(img, dsize=(960, 640), fx=1, fy=1)
        cv2.imwrite('./static/mp4_result.jpg', img)

    # 释放视频对象
    video.release()
    return '1'
@app.route('/get_image', methods=['GET'])
def get_image():
    image_path = './static/mp4_result.jpg'
    with open('./static/mp4_result.jpg', 'rb') as img_f:
        img_stream = img_f.read()
        print(type(img_stream))
        img_stream = base64.b64encode(img_stream).decode()
        print(type(img_stream))
    #return render_template('flask_web.html',img_stream)
    return img_stream
    #return jsonify(image_path)


if __name__ == '__main__':
    app.run(debug=True)
