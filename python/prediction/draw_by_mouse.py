import cv2
import numpy as np

point_list = []
point = (-1, -1)
mode = 0


def info(_img):
    x, y = 50, 80
    info_str1 = 'key1:select point'
    info_str2 = 'key2:draw shape'
    info_str3 = 'key3:draw line'
    info_str4 = 'key0:clear all point'
    cv2.putText(_img, info_str1, (x, y), 0, 2, (255, 255, 0), 4)
    cv2.putText(_img, info_str2, (x, y+80), 0, 2, (255, 255, 0), 4)
    cv2.putText(_img, info_str3, (x, y+160), 0, 2, (255, 255, 0), 4)
    cv2.putText(_img, info_str4, (x, y+240), 0, 2, (255, 255, 0), 4)
    return _img


def draw_img(event, x, y, flags, param):
    """
            鼠标响应
        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        """
    global point_list, point
    global img, img_ups
    if event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动，提示信息
        img_text = img_ups.copy()
        cv2.putText(img_text, f'({x}, {y})', (x, y), 0, 2, (0, 255, 0), 4)
        cv2.imshow('pic1', img_text)
    # 左键选择点
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        # 展示
        cv2.circle(img_ups, point, 5, (255, 0, 255), 10)
        cv2.imshow('pic1', img_ups)
        # 添加点
        point_list.append(point)
    # 右键取消上一点
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 大于1个点
        if len(point_list) > 0:
            # 删除重置
            point_list.pop()
            img_ups = img.copy()
        # 删除之后还大于1个点
        if len(point_list) > 0:
            # 重置
            img_ups = img.copy()
            for p in point_list:
                cv2.circle(img_ups, p, 5, (255, 0, 255), 10)
        cv2.imshow('pic1', img_ups)


def draw_shape():
    """
        画多边形
    """
    global point_list, mode
    mode = 1
    surround_point = np.array([point_list], dtype=np.int32)
    if len(point_list) < 4:
        print('error, not enough point')
    else:
        drawing_img = cv2.polylines(img_ups, surround_point, True, (0, 255, 255), 3)
        cv2.imshow('pic1', drawing_img)


def output_file():##输出坐标到文档
    global point_list
    with open('point.txt', 'a', encoding="utf-8") as file:
        if mode == 1:##输出多边形
            file.write('shape:'+'\n')
            for _ in point_list:
                file.write(str(_)+'\n')
            file.write('end'+'\n')
        elif mode == 3:
            file.write('line'+'\n')
            for _ in point_list:
                file.write(str(_)+'\n')
            file.write('end'+'\n')


def draw_line():
    """
        绘制直线
    """
    global point_list, mode
    mode = 3
    if len(point_list) == 2:
        cv2.line(img_ups, point_list[0], point_list[1], (155, 255, 155), 2)
    else:
        print('error, point number error')


if __name__ == '__main__':
    # 加载视频
    video = cv2.VideoCapture("bigroad.mp4")
    # 读取第一帧视频
    ret, img = video.read()
    if not video.isOpened():
        print('Video error, load error')
    # 展示用图像
    img = info(img)
    img_ups = img.copy()

    img_y, img_x = img.shape[:2]#图像宽，高

    # 展示第一帧图像
    cv2.namedWindow('pic1', cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('pic1', draw_img)

    # 选择操作
    print('选择操作', end='\n')

    while True:
        cv2.imshow('pic1', img_ups)
        key = cv2.waitKey(100)
        if key == 27:###esc退出
            break

        # 选择操作
        elif key == 49:#按键1绘制多边形
            draw_shape()
        elif key == 50:#按键2输出
            output_file()
        elif key == 51:#按键3切换绘制直线
            draw_line()
        elif key == 48:#按键0清空
            point_list.clear()
            img_ups = img.copy()
            cv2.imshow('pic1', img_ups)
