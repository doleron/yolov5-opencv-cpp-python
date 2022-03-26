import random

import cv2
import numpy as np

# configs
video_path = '../Videos/bigroad.mp4'
onnx_file = '../config_files/best.onnx'
class_file = '../config_files/classes.txt'
out = cv2.VideoWriter('output.flv', cv2.VideoWriter_fourcc('F', 'L', 'V', '1'), 30, (1600, 1600))
is_cuda = False if cv2.cuda.getCudaEnabledDeviceCount() != 1 else True
# --------------------------------
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

net = cv2.dnn.readNet(onnx_file)

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]


def load_classes():
    class_list = []
    with open(class_file, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


class_list = load_classes()


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def wrap_detection(input_image, output_data):
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
        if confidence.any() >= 0.4:

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




def main():

    if is_cuda:
        print("Attempt to use CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("Attempt to use Pure CPU...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # 定义摄像机
    capture = cv2.VideoCapture(video_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Detecting...")
    while True:

        process(capture)

        if cv2.waitKey(1) > -1:
            print("Finished by user")
            cv2.destroyAllWindows()
            break

    capture.release()
    cv2.destroyAllWindows()


def process(capture):
    _, frame = capture.read()
    frame = format_yolov5(frame)

    (h, w) = frame.shape[:2]
    width = 1600
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward()
    class_ids, confidences, boxes = wrap_detection(frame, outs[0])

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))


    cv2.imshow("output", frame)


if __name__ == '__main__':
    main()
