import random
import threading

import cv2
import time
import sys
import numpy as np
import webRtc
from PIL import Image
import copy
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend


root = ""
root = "/home/aieson/Code/A01/yolov5-opencv-cpp-python"

# Configs
video_path = root + "/Videos/bigroad.mp4"
onnx_path = root + "/config_files/best.onnx"
class_file_path = "/config_files/classes.txt"

# onnx_path = root + "/config_files/yolov5s.onnx"
# class_file_path = "/config_files/classes_yolov5s.txt"
is_cuda = False if cv2.cuda.getCudaEnabledDeviceCount() == 0 else True


# 鼠标选定区域
banned_areas = []
window_name = "output"


pts = [deque(maxlen=30) for _ in range(9999)]



def build_model(is_cuda):
    net = cv2.dnn.readNet(onnx_path)
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.1


def detect(image, net):
    cv2.setNumThreads(16)
    # print("Detecting...")
    start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

    net.setInput(blob)
    preds = net.forward()
    end = time.time()
    print((end - start) * 1000, "ms")
    return preds

def load_capture():
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    return capture

def load_classes():
    class_list = []
    with open(root + class_file_path,  "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()


def wrap_detection(input_image, output_data):
    cv2.setNumThreads(16)
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT



    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.1:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    # frame = frame.get()
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

net = build_model(is_cuda)
capture = load_capture()


start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))



# get_area_from_mouse()

# 红绿灯区域

lights = np.array([
    [
        [787, 132],
        [802, 132],
        [802, 138],
        [787, 138]
    ],
    [
        [805, 132],
        [819, 132],
        [819, 138],
        [805, 138]
    ]

])

# 先手动制定一下区域
banned_areas = np.array([
    [88, 465],
    [532, 470],
    [275, 786],
    [0, 660],
    [0, 506]
                         ], np.int32)
banned_areas = banned_areas.reshape((-1, 1, 2))

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
gst_str_rtp = "appsrc ! videoconvert ! video/x-raw,format=I420,width=1280,height=720,framerate=15/1 ! x264enc speed-preset=superfast tune=zerolatency ! rtph264pay ! udpsink host=1.116.157.189 port=8004"
out = cv2.VideoWriter(gst_str_rtp, 0, 15, (1280, 720), True)

counter = []

# deepsort
model_filename = '/home/aieson/Code/A01/yolov5-opencv-cpp-python/python/dpsr/Object-Detection-and-Tracking-using-YOLOv3-and-DeepSort/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', 0.5, None)
tracker = Tracker(metric)


while True:
    _, frame = capture.read()

    if frame is None:
        print("End of stream")
        break

    # myThread(frame).start()
    (h, w) = frame.shape[:2]
    width = 1600
    r = width / float(w)
    dim = (width, int(h * r))

    # frame = cv2.UMat(frame)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])


    image = Image.fromarray(frame[..., ::-1])

    features = encoder(frame, boxes)

    detections = [Detection(bbox, 1.0, class_list, feature) for bbox, feature in zip(boxes, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, 0.3, confidences[0])
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    i = int(0)
    indexIDs = []
    c = []
    boxes = []
    for det in detections:
        bbox = det.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        # boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track.track_id))
        counter.append(int(track.track_id))
        bbox = track.to_tlbr()
        color = [int(c) for c in colors[indexIDs[i] % len(colors)]]

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
        cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)
        if len(class_list) > 0:
            class_name = class_list[0]
            cv2.putText(frame, str(class_list[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)

        i += 1
        # bbox_center_point(x,y)
        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        # track_id[center]
        pts[track.track_id].append(center)
        thickness = 5
        # center point
        cv2.circle(frame, (center), 1, color, thickness)

        # draw motion path
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
            # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

    # frame_count += 1
    # total_frames += 1

    for light in lights:
        cv2.polylines(frame, [light], True, colors[2], 1)

        overlay = frame.copy()
        output = frame.copy()

        cv2.fillPoly(overlay, [light], colors[2])
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



    # 
    # 
    # 
    # circle(frame, (box[0] + (box[2] // 2), box[1] + (box[3] // 2)), radius=10, color=color[1])

    # if frame_count >= 30:
    #     end = time.time_ns()
    #     fps = 1000000000 * frame_count / (end - start)
    #     frame_count = 0
    #     start = time.time_ns()

    if fps > 0:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

    # frame = frame.get()
    cv2.polylines(frame, [banned_areas], True, colors[2], 5)

    overlay = frame.copy()
    output = frame.copy()

    cv2.fillPoly(overlay, [banned_areas], colors[2])
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # print("Writing frame...")
    # out.write(frame)
    # print("Done")

    cv2.imshow(window_name, frame)


    # webRtc.wrapvideo(frame, _)
    if cv2.waitKey(0) & 0xFF == ord('q'): # 按下 q
        print("finished by user")
        cv2.destroyAllWindows()
        break



print("Total frames: " + str(total_frames))

if __name__ == "__main__":
    pass
