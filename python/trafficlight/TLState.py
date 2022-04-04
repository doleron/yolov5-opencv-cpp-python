import os

import cv2
import random
import numpy as np
from enum import Enum
from detectColor import detectColor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class TLState(Enum):
    none = 0
    red = 1
    yellow = 2
    green = 3
    red_yellowArrow = 4
    red_greenArrow = 5
    green_yellowArrow = 6
    green_greenArrow = 7
    redArrow = 8
    yellowArrow = 9
    greenArrow = 10
    flashingYellowArrow = 11

class TLType(Enum):
    regular = 0
    five_lights = 1
    four_lights = 2

def imgResize(image, height, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # calculate the ratio of the height and construct the dimensions
    r = height / float(h)
    dim = (int(w * r), height)
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def detectState(image, TLType):
    image = imgResize(image, 200)
    (height, width) = image.shape[:2]
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 霍夫圆环检测
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=15,maxRadius=30)
    overallState = 0
    stateArrow = 0
    stateSolid = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            if i[1] < i[2]:
                i[1] = i[2]
            roi = image[(i[1]-i[2]):(i[1]+i[2]),(i[0]-i[2]):(i[0]+i[2])]
            color = detectColor(roi)
            if color > 0:
                if TLType == 1 and i[0] < width/2 and i[1] > height/3:
                    stateArrow = color
                elif TLType == 2:
                    stateArrow = color
                    if i[1] > height/2 and i[1] < height/4*3:
                        stateArrow = color + 2
                else:
                    stateSolid = color

    if TLType == 1:
        overallState = stateArrow + stateSolid + 1
    elif TLType == 2:
        overallState = stateArrow + 7
    else:
        overallState = stateSolid

    return overallState

def plot_light_result(images):

    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        lena = mpimg.imread(image)
        label = TLState(detectState(cv2.imread(image),TLType.regular.value)).name
        plt.title(label)
        plt.imshow(lena)
    plt.show()




light_path = []
for i in os.listdir('imgs/norm'):
    light_path.append('imgs/norm/' + i)
print(light_path)
random.shuffle(light_path)
plot_light_result(light_path)

def plot_arrow_result(images):

    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        lena = mpimg.imread(image)
        label = TLState(detectState(cv2.imread(image),TLType.five_lights.value)).name
        plt.title(label)
        plt.imshow(imgResize(lena, 200))
    plt.show()

arrow_path = []
for i in os.listdir('imgs/arr'):
    arrow_path.append('imgs/arr/' + i)
random.shuffle(arrow_path)
plot_arrow_result(arrow_path)