from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter2 import draw_up_down_counter
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
import sys
from hyperlpr import *
from PIL import Image, ImageDraw, ImageFont
import numpy


sys.path.insert(0, './yolov5')


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def counter_vehicles(outputs, line_pixel, dividing_pixel, counter_recording,up_counter,down_counter):
    box_centers = []
    box_w = []
    for i, each_box in enumerate(outputs):
        ###求得每个框的中心点
        box_centers.append([(each_box[0] + each_box[2]) / 2, (each_box[1] + each_box[3]) / 2, each_box[4],
                             each_box[5],each_box[2]-each_box[0]])

    for box_center in box_centers:
        id_recorded = False
        if len(counter_recording)==0:
            if box_center[0] <= dividing_pixel and box_center[1] >= line_pixel:
                down_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
            elif box_center[0] > dividing_pixel and box_center[1] < line_pixel:
                up_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
        if len(counter_recording)>0:
            for n in counter_recording:
                if n == box_center[2]:  ###判断该车辆是否已经记过数
                    id_recorded = True
                    break
            if id_recorded:
                continue
            if box_center[0] <= dividing_pixel and box_center[1] >= line_pixel:
                down_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
            elif box_center[0] > dividing_pixel and box_center[1] < line_pixel:
                up_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue

    return counter_recording, up_counter, down_counter, box_centers

def Estimated_speed(locations, fps,width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index=[]
    work_IDs_prev_index=[]
    work_locations=[]
    work_prev_locations = []
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])### 获得当前帧中跟踪到车辆的ID及对应的类型
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(math.sqrt((work_locations[i][0] - work_prev_locations[i][0])**2+
                               (work_locations[i][1]- work_prev_locations[i][1])**2)*
                     width[work_locations[i][3]]/ (work_locations[i][4])*fps/5*3.6*2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i],1),work_locations[i][2]]
    return speed

def draw_speed(img, speed, bbox_xywh, identities):
    for i,j in enumerate(speed):
        for m, n in enumerate(identities):
            if j[1]==n:
                xy = [int(i) for i in bbox_xywh[m]]
                cv2.putText(img, 'speed:'+ str(j[0])+'km/h', (xy[0], xy[1]-7), cv2.FONT_HERSHEY_PLAIN,1.5, [255, 255, 255], 2)
                break
def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_over_speed(img, over_speed_id, plates_record):
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('font/simsun.ttc', 40)  # 加载字体
    draw = ImageDraw.Draw(img_PIL)
    for nu, each_id in enumerate(over_speed_id):
        label = '%s over speed' %plates_record[str(each_id)]
        draw.text((10,100+nu*20), label, font=font,fill=(255, 0, 0))
    img = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)


def draw_boxes(img, bbox, cls_names, classes2,over_speed_id, identities=None):
    offset = (0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        if id in over_speed_id:
            color = [0,0,255]
        else:
            color = compute_color_for_labels(int(classes2[i]*100))
        label = '%d %s' % (id, cls_names[i])
        #label +='%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    ##获得视频的帧宽高
    capture = cv2.VideoCapture(source)
    frame_fature = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    names = ['car','truck','bus']
    class_name = dict(zip(list(range(len(names))), names))

    ###设置计数器
    counter_recording = []
    up_counter = [0]*len(names)
    down_counter = [0]*len(names)
    line_pixel = [frame_fature[1]//2]
    dividing_pixel = frame_fature[0]
    ###设置每种车型的车宽
    width = [1.85, 2.3, 2.5]
    #各参数初始化
    locations=[]
    speed=[]
    plates_record={}
    speed_limit=15
    over_speed_id=[]
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'


    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        print(pred)
        # Apply NMS(非极大值抑制)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size(用scale_coords函数来将图像缩放)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                classes = []
                box_xywh=[]
                img_h, img_w, _ = im0.shape
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:

                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0, classes)
                #返回计数器和id记录
                counter_recording, up_counter, down_counter, location = counter_vehicles(outputs, line_pixel, dividing_pixel, counter_recording, up_counter, down_counter)
                locations.append(location)
                #每5帧对视频中的目标测一次速度
                if len(locations)==5:
                    if len(locations[0]) and len(locations[-1]) !=0:
                        locations=[locations[0],locations[-1]]
                        speed = Estimated_speed(locations, fps, width)
                    locations=[]
                ###判断超速
                id_pl_record=[]
                over_speed_id=[]
                for each_speed in speed:
                    if each_speed[0]>speed_limit:
                        over_speed_id.append(each_speed[1])
                now_id = []
                for each_output in outputs:
                    now_id.append(each_output[4])
                del_index=[]
                for nu,each_s_id in enumerate(over_speed_id):
                    if each_s_id not in now_id:
                        del_index.append(nu)
                if len(del_index):
                    for del_i in del_index[::-1]:
                        del over_speed_id[del_i]
                for each_plate in plates_record:
                    id_pl_record.append(each_plate[0])
                #绘制车牌
                for each_output in outputs:
                    #绘制中文字符时需要对cv格式的图像进行转换
                    img_PIL = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                    font = ImageFont.truetype('font/simsun.ttc', 40) #加载字体
                    # 截取检测框范围内的图像
                    cropped=im0[int(each_output[1]):int(each_output[3]),int(each_output[0]):int(each_output[2])]
                    #对截取检测框内的图像进行车牌识别
                    plate = HyperLPR_plate_recognition(cropped)

                    #择优更新车牌记录中的信息
                    if len(plate):
                        draw = ImageDraw.Draw(img_PIL)
                        if str(each_output[4]) in plates_record:
                            chepai = plates_record[str(each_output[4])]
                            if chepai[1]<plate[0][1]:
                                plates_record[str(each_output[4])]=[plate[0][0], plate[0][1]]
                                draw.text((each_output[0], each_output[1] + 30), plates_record[str(each_output[4])][0],font=font, fill=(255, 0, 0))
                            else:
                                draw.text((each_output[0], each_output[1] + 30), plates_record[str(each_output[4])][0],font=font, fill=(255, 0, 0))
                        else:
                            plates_record[str(each_output[4])] = [plate[0][0], plate[0][1]]
                            draw.text((each_output[0], each_output[1] + 30), plates_record[str(each_output[4])][0],font=font, fill=(255, 0, 0))
                        im0 = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(im0,(each_output[0]+plate[0][2][0]+5,each_output[1]+plate[0][2][1]+7),
                                      (each_output[0]+plate[0][2][2]+5,each_output[1]+plate[0][2][3]+8),(0,0,255),thickness=3)
                        #cv2.putText(im0,plate[0][0],(each_output[0],each_output[1]),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)

                for each_id in over_speed_id:###将超速的车牌写入文档中
                    with open('over_speed_record.txt','a',encoding="utf-8") as file:
                        for each_speed in speed:
                            if each_id==each_speed[1]:
                                file.write(plates_record[str(each_id)][0]+':'+str(each_speed[0])+'km/h'+'\n')

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    classes2 = outputs[:, -1]

                    draw_speed(im0, speed, bbox_xyxy, identities)
                    #draw_over_speed(im0, over_speed_id, plates_record)
                    img_PIL = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                    font = ImageFont.truetype('font/simsun.ttc', 40)  # 加载字体
                    draw = ImageDraw.Draw(img_PIL)
                    for nu, each_id in enumerate(over_speed_id):
                        label = '%s over speed' % plates_record[str(each_id)][0]
                        draw.text((10, 100 + nu * 40), label, font=font, fill=(255, 0, 0))
                    im0 = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)

                    draw_boxes(im0, bbox_xyxy, [names[i] for i in classes2], classes2, over_speed_id,identities)
                    draw_up_down_counter(im0, up_counter, down_counter,frame_fature, names)
                    cv2.line(im0, (0, frame_fature[1]//2), (frame_fature[0], frame_fature[1]//2),(0, 0, 255), 2)
                    cv2.line(im0, (dividing_pixel,0),(dividing_pixel,frame_fature[1]),(0,255,0),2)
                    ###
                    if len(over_speed_id):
                        label1=''
                        for i in over_speed_id:
                            label1 =  label1+plates_record[str(i)][0]+' '
                            label1=str(label1)
                        cv2.imencode('.jpg', im0)[1].tofile('inference/images/%s.jpg' % str(label1))
                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:  
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results

            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
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

    parser.add_argument('--classes', nargs='+', type=int, default=[0,1,2], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
