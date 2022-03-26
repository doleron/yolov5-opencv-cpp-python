#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :test2.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/8/2
@Desc   :
=================================================='''
import argparse

import cv2

print(cv2.cuda.getCudaEnabledDeviceCount())

import numpy as np
import time

from tqdm import tqdm
import os
from pathlib import Path


class yolov5():
    def __init__(self, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        # with open('coco.names', 'rt') as f:
        #     self.classes = f.read().rstrip('\n').split(
        #         '\n')  ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.classes = ['dent', 'scratch', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                        'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                        'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                        'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                        'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                        'teddy bear',
                        'hair drier', 'toothbrush']

        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNetFromONNX('/home/aieson/Downloads/yolov5s.onnx')
        # 一下两行代码是设置dnn cuda加速配置可选
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # 求缩放比例
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence) * float(detection[4]))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        # NMS非极大值抑制算法去掉重复的框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]  # 取第一个
            box = boxes[i]  # 取出检测框
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    def sigmoid(self, inX):
        from numpy import exp
        if inX >= 0:
            return 1.0 / (1 + exp(-inX))
        else:
            return exp(inX) / (1 + exp(inX))

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True,
                                     crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        outs = 1 / (1 + np.exp(-outs))  ###定义sigmoid函数
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                self.grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs


def mult_test(yolov5, img_dir):
    """
    批量测试yolov5
    :param yolov5: 模型
    :param img_dir: 文件夹
    :return:
    """
    path = Path(img_dir)

    for file in tqdm(os.listdir(img_dir)):
        start = time.time()  # 记录开始时间
        file_name = Path(file).stem  # 获取基本名字
        srcimg = cv2.imread(str(path / file))
        dets = yolov5.detect(srcimg)
        srcimg = yolov5.postprocess(srcimg, dets)
        cost = time.time() - start
        cost = round(cost, 2)
        cv2.putText(srcimg, "cost time:" + str(cost), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        cv2.imwrite(str(path / file_name) + "_reslut.png", srcimg)


if __name__ == "__main__":

    yolonet = yolov5(confThreshold=0.25, nmsThreshold=0.35,
                     objThreshold=0.2)
    srcimg = cv2.imread('a.png')
    dets = yolonet.detect(srcimg)
    srcimg = yolonet.postprocess(srcimg, dets)
    cv2.imshow("output", srcimg)
    if cv2.waitKey() > -1:
        print("finished by user")
        cv2.destroyAllWindows()

    # img_dir = r'D:\Projects\magic\origin_coco'
    # img_dir = r'D:\Projects\test_yolo'
    # mult_test(yolonet, img_dir)
    # start = time.time()  # 记录开始时间
    # srcimg = cv2.imread(args.imgpath)
    #
    # dets = yolonet.detect(srcimg)
    # srcimg = yolonet.postprocess(srcimg, dets)
    # cost = time.time() - start
    # cost = round(cost, 2)
    # print(f"cost time is {cost}")
    # cv2.putText(srcimg, "cost time:" + str(cost), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    #
    # winName = 'Deep learning object detection in OpenCV'
    # cv2.namedWindow(winName, 0)
    # cv2.imshow(winName, srcimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # vid = cv2.VideoCapture(0)
    # count = 0
    # xun = 0
    # while True:
    #     _, frame = vid.read()
    #     dets = yolonet.detect(frame)
    #     srcimg = yolonet.postprocess(frame, dets)
    #     # if count == 0:
    #     #     dets = yolonet.detect(frame)
    #     #     srcimg = yolonet.postprocess(frame, dets)
    #     #     count += 1
    #     # else:
    #     #     count = (count + 1) % 10
    #
    #     cv2.imshow("result", srcimg)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # vid.release()
    # cv2.destroyAllWindows()