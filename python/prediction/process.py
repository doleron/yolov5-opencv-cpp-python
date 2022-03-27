import cv2
from counter.draw_counter2 import draw_up_down_counter
from track_copy import *
import time


def record_time():
    """
        获得当前时间 return str型 or 年月日时分秒
    """
    times = time.localtime()
    year, mon, day, hour, minn, sec, *_ = times
    time_str = str(year) + "年" + str(mon) + "月" + str(day) + "日" +\
                str(hour) + "时" + str(minn) + "分" + str(sec) + "秒"
    return time_str


def parser_prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='source.avi', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')

    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    return args


if __name__ == '__main__':

    print(record_time())

    args = parser_prepare()
    with torch.no_grad():
        detect(args)
