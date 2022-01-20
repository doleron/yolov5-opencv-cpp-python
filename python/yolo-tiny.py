import numpy as np
import cv2

# step 1 - load the model

net = cv2.dnn.readNet('config_files/yolov5s.onnx')

# step 2 - feed a 640x640 image to get predictions

image = cv2.imread('misc/car.jpg')
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True)
net.setInput(blob)
predictions = net.forward()

# step 3 - unwrap the predictions to get the object detections 

class_ids = []
confidences = []
boxes = []

output_data = predictions[0]

for r in range(25200):
    row = output_data[r]
    confidence = row[4]
    if confidence >= 0.4:

        classes_scores = row[5:]
        _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
        class_id = max_indx[1]
        if (classes_scores[class_id] > .25):

            confidences.append(confidence)

            class_ids.append(class_id)

            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
            left = int(x - 0.5 * w)
            top = int(y - 0.5 * h)
            box = np.array([left, top, int(w), int(h)])
            boxes.append(box)

class_list = []
with open("config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

for i in range(len(class_ids)):

    box = boxes[i]
    class_id = class_ids[i]

    cv2.rectangle(image, box, (0, 255, 255), 2)
    cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
    cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

cv2.imshow("output", image)
cv2.waitKey()
