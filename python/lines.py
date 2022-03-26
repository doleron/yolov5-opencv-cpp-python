import cv2


class Lines:
    def __init__(self, n, fitl):
        self.Name = n
        self.FitL = fitl


class Lanes:
    def __init__(self, go, n, fitl, fitr):
        self.go = go
        self.Name = n
        self.FitL = fitl
        self.FitR = fitr


class StopArea:
    pass


class StopLine:
    pass

#------------------------------------------#
def creat_line(name, l, r):
    return Lines(name, l, r)


def trans_lane(go, n, fitl, fitr):
    return Lanes(go, n, fitl, fitr)


if __name__ == '__main__':
    pass