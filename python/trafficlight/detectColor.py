import cv2
import numpy as np


cv2.dnn.ca

def detectColor(image): # receives an ROI containing a single light
    # convert BGR image to HSV
    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # min and max HSV values
    red_min = np.array([0,5,150])
    red_max = np.array([8,255,255])
    red_min2 = np.array([175,5,150])
    red_max2 = np.array([180,255,255])

    yellow_min = np.array([20,5,150])
    yellow_max = np.array([30,255,255])

    green_min = np.array([35,5,150])
    green_max = np.array([90,255,255])

    # apply red, yellow, green thresh to image
    # 利用cv2.inRange函数设阈值，去除背景部分
    red_thresh = cv2.inRange(hsv_img,red_min,red_max)+cv2.inRange(hsv_img,red_min2,red_max2)
    yellow_thresh = cv2.inRange(hsv_img,yellow_min,yellow_max)
    green_thresh = cv2.inRange(hsv_img,green_min,green_max)

    # apply blur to fix noise in thresh
    # 进行中值滤波
    red_blur = cv2.medianBlur(red_thresh,5)
    yellow_blur = cv2.medianBlur(yellow_thresh,5)
    green_blur = cv2.medianBlur(green_thresh,5)

    # checks which colour thresh has the most white pixels
    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yellow_blur)
    green = cv2.countNonZero(green_blur)

    # the state of the light is the one with the greatest number of white pixels
    lightColor = max(red,yellow,green)

    # pixel count must be greater than 60 to be a valid colour state (solid light or arrow)
    # since the ROI is a rectangle that includes a small area around the circle
    # which can be detected as yellow
    if lightColor > 60:
        if lightColor == red:
            return 1
        elif lightColor == yellow:
            return 2
        elif lightColor == green:
            return 3
    else:
        return 0