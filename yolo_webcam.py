from ultralytics import YOLO
import cv2
import numpy as np
import cvzone # used to display detections
import math
import argparse
from sort import *

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# tracker
tracker = Sort(max_age=20, min_hits=3,iou_threshold=0.3)

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#     help="path to input image")
# ap.add_argument("-y", "--yolo", required=True,
#     help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
#     help="threshold when applying non-maxima suppression")
# args = vars(ap.parse_args())

# cap = cv2.VideoCapture(0) # For webcam
cap = cv2.VideoCapture("Videos/stopped_vehicle_assignment.avi")
cap.set(3,658) # width
cap.set(4,420) # height

model = YOLO("Yolo-Weights/yolov8m.pt")
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

frame_num = 0
positionTracker = {}

while True:
    success, img = cap.read()
    results = model(img,stream=True)

    detections = []
    frame_num+=1

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w, h = x2-x1,y2-y1

            conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "motorbike", "bus", "truck"] and conf > 0.3:
                currentArray = [x1,y1,x2,y2,conf]
                detections.append(currentArray)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
                # cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2, offset=10)

    resultsTracker = tracker.update(np.array(detections))

    for result in resultsTracker:
        x1,y1,x2,y2,id = map(int, result)
        currentPosition = [x1,y1,x2,y2]
        w, h = x2 - x1, y2 - y1
        print(f"Current position: {id} {currentPosition}")
        print(f"positionTracker: {positionTracker}")
        if id not in positionTracker:
            positionTracker[id] = currentPosition
        else:
            previousPosition = positionTracker[id]
            distance = math.sqrt((previousPosition[0] - x1) ** 2 + (previousPosition[1] - y1) ** 2)
            if (x1<previousPosition[0] + 20 and y1<previousPosition[1] + 20 and
                x2<previousPosition[2] + 20 and y2<previousPosition[3] + 20) or distance <= 10 :
                positionTracker[id] = currentPosition
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
                cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=2, offset=10)


    ids_to_remove = []
    for id, currentPosition in positionTracker.items():
        dist1 = math.sqrt((currentPosition[0] - x1) ** 2 + (currentPosition[1] - y1) ** 2)
        dist2 = math.sqrt(
            (currentPosition[2] - currentPosition[0]) ** 2 + (currentPosition[3] - currentPosition[1]) ** 2)

        if dist2 >= dist1 or dist1 > 600:
            ids_to_remove.append(id)


    # Remove the vehicles with IDs in ids_to_remove
    for id in ids_to_remove:
        del positionTracker[id]



                # if row[0] == id and (x1>=row[1] or y1>=row[2] or
                #         x2>=row[3] or y2>=row[4]):
                #     # print(f"row[0]: {row[0]} and id: {id}")
                #     if x1 < row[1] + 10 and y1 < row[2] + 10 and \
                #         x2 < row[3] + 10 and y2 < row[4] + 10:
                #         cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=2,colorR=(255,0,0))
                #         cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)),
                #                            scale=2, thickness=2, offset=10)
                #     else:
                #         row[1],row[2],row[3],row[4] = x1,y1,x2,y2



        # if id in positionTracker:

        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        # cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=2, offset=10)



    if img.shape[0] != 0 and img.shape[1] !=0:
        cv2.imshow("Image",img)
    cv2.waitKey(1) # 1ms delay



