import cv2
import numpy as np

#所有车辆字典
#car_dic = {}

#车辆ID：【位置[close_l][close_r][far_l][far_r]】【方向=close or far】


class Car:
    """
        车辆实例化字典
        #key: plate -> ...
        #key: bbox -> [(), (), (), ()]
        #key: core -> ()
        #key: direction -> 'up' or 'down' // 车道才有用
        #key: lane -> 'cd1' or... // 车道
        #key: count -> ...
        #key: violations -> {all violation report : [0, 'state']}
    """
    def __init__(self):
        """
            初始化该车辆字典
        """
        self.car_dic = {'count': 0, 'violations': {}, 'bbox': [(), (), (), ()], 'core': self.get_core()}

    def __setitem__(self, key, value):
        """
            添加或更改数据
        """
        self.car_dic[key] = value

    def __getitem__(self, item):
        """
            返回相应的value值
        """
        return self.car_dic.get(item, 0)

    def __len__(self):
        """
            返回key的长度
        """
        return len(self.car_dic)

    def get_core(self, bbox='bbox'):
        """
            得到车辆中心
        """
        core_x = self.car_dic[bbox][0][0] + self.car_dic[bbox][1][0] + \
                 self.car_dic[bbox][2][0] + self.car_dic[bbox][3][0] // 4
        core_y = self.car_dic[bbox][0][1] + self.car_dic[bbox][1][1] + \
                 self.car_dic[bbox][2][1] + self.car_dic[bbox][3][1] // 4
        return core_x, core_y

    def add_problem(self, _problem):
        """
            添加违规情况，
        """
        self.car_dic['problem'].append(_problem)

    def get_all_problem(self):
        """
            得到所有违规情况
        """
        for problem in self.car_dic['problems']:
            pass


if __name__ == '__main__':
    pass
