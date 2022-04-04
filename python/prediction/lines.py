import cv2
import numpy as np


#生成车道字典-------------------------------------------------
# for number in range(1, chedao_num + 1):
#     #生成车道名字
#     first_name = 'cd'
#     name = first_name + f'{number}'
#     #生成车道空字典
#     lines_dic[name] = {}
#     #组装左右车道线
#     lines_dic[name]['L'] = 1
#     lines_dic[name]['R'] = 2
#     #组装车道方向 far or close or no
#     lines_dic[name]['direction'] = 'far'
#     #组装车道位置模块 true 可通行， false 不可通行
#     lines_dic[name]['mod'] = {'L': True, 'R': False}

#线的拟合方式：暂时用两点确定一条直线
#得到X=KY+B


class Line:
    """
        线段实例化
    """
    def __init__(self, n, point1, point2):
        self.name = n
        self.fy, self.fy_k, self.fy_b = self.get_fit(point1, point2)
        self.point = [point1, point2]

    def get_name(self):
        """
            得到线的名字
        """
        return self.name

    def get_minmaxx(self):
        """
            返回最小,最大的x
        """
        minx = min(self.point, key=lambda x: x[0])
        maxx = max(self.point, key=lambda x: x[0])
        if maxx - minx < 200: #保证最小有200像素点
            _ = (200 - maxx + minx) // 2
            minx -= _
            maxx += _
        return minx, maxx

    def get_minmaxy(self):
        """
            返回最小，最大的y
        """
        miny = min(self.point, key=lambda y: y[1])
        maxy = max(self.point, key=lambda y: y[1])
        if maxy - miny < 200: #保证最小有200像素点
            _ = (200 - maxy + miny) // 2
            miny -= _
            maxy += _
        return miny, maxy

    def fy_to_x(self, y):
        return self.fy(y)

    @classmethod
    def get_fit(cls, point1, point2):
        """
            得到一次直线拟合
        """
        x = (point1[0], point2[0])
        y = (point1[1], point2[1])
        f = np.polyfit(y, x, 1)
        fy = np.poly1d(f)
        fy_k = fy[1]
        fy_b = fy[0]
        return fy, fy_k, fy_b









class Lane:
    def __init__(self, go, n, fitl, fitr):
        self.go = go
        self.Name = n
        self.FitL = fitl
        self.FitR = fitr

    def is_position(self):
        l_function = self.FitL
        r_function = self.FitR



class StopArea:
    pass


class StopLine:
    pass

#------------------------------------------#


def creat_line(name, l, r):
    pass
   # return Lines(name, l, r)


def trans_lane(go, n, fitl, fitr):
    pass
    #return Lanes(go, n, fitl, fitr)


if __name__ == '__main__':
    pass
