import cv2
import numpy as np

point_list = []
point = (-1, -1)


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
        cv2.putText(img_text, f'({x}, {y})', (x, y), 0, 2, (255, 255, 0), 4)
        cv2.circle(img_text, point, 5, (255, 150, 255), 10)
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



def run1():
    """
        画多边形
    """
    global point_list
    surround_point = np.array([point_list], dtype=np.int32)
    if len(point_list) < 4:
        print('error, not enough point')
    else:
        quxing_img = cv2.polylines(img_ups, surround_point, True, (255, 155, 255), 2)
        cv2.imshow('pic1', quxing_img)


def output_file():
    global point_list
    with open('point.txt', 'a', encoding="utf-8") as file:
        for _ in point_list:
            file.write(str(_)+'\n')
        file.write('end'+'\n')


if __name__ == '__main__':
    # 加载视频
    video = cv2.VideoCapture("bigroad.mp4")
    # 读取第一帧视频
    ret, img = video.read()
    if not video.isOpened():
        print('Video error, load error')
    # 图像备份
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
            run1()

        elif key == 50:#按键2输出
            output_file()


