from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import serial
from sort import *
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Thread, Timer
from multiprocessing import Process, Manager
import time

app = Flask(__name__)
CORS(app)
motorbike_count = 0
car_count = 0

@app.route('/api/motorbike', methods=['GET'])
def get_motorbikeCount():
    global motorbike_count
    return jsonify({'motorbike_count': motorbike_count})

@app.route('/api/car', methods=['GET'])
def get_carCount():
    global car_count
    return jsonify({'car_count': car_count})

def update_motorbike_count(count):
    global motorbike_count
    motorbike_count = count

def update_car_count(count):
    global car_count
    car_count = count

def run_flask_app():
    app.run(debug=True, use_reloader=False, port=8080)

flask_thread = Thread(target=run_flask_app)
flask_thread.start()

cap = cv2.VideoCapture('asset/videos/pradita-vehicle-counting.mp4') # For Video
# cap.set(4, 720)
# cap.set(3, 1280)

model = YOLO("yolov8n.pt")

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

mask = cv2.imread("asset/images/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsIn = [826, 666, 1378, 634]

totalCount = []
carCount = []
motorbikeCount = []

# Define the ESP32 serial port and baud rate
serial_port = 'COM5'
baud_rate = 9600

# Open serial connection to ESP32
ser = serial.Serial(serial_port, baud_rate)
time.sleep(2)

def control_led(color):
    if color == 'green':
        ser.write(b'G')
    elif color == 'red':
        ser.write(b'R')
    elif color == 'motorbike':
        ser.write(b'M')
    elif color == 'car':
        ser.write(b'C')

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    carGraphics = cv2.imread("asset/images/car-graphics.png", cv2.IMREAD_UNCHANGED)
    motorbikeGraphics = cv2.imread("asset/images/motorbike-graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, carGraphics, (0, 0))
    img = cvzone.overlayPNG(img, motorbikeGraphics, (0, 162))

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limitsIn[0], limitsIn[1]), (limitsIn[2], limitsIn[3]), (0, 0, 255), thickness=5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsIn[0] < cx < limitsIn[2] and limitsIn[1] - 20 < cy < limitsIn[1] + 20:
            if cls == 2 or cls == 7:
                if id not in carCount:
                    carCount.append(id)
            elif cls == 3:  # Motorbike
                if id not in motorbikeCount:
                    motorbikeCount.append(id)
            cv2.line(img, (limitsIn[0], limitsIn[1]), (limitsIn[2], limitsIn[3]), (0, 255, 0), thickness=5)

    # Control LED based on motorbikeCount and carCount
    if len(motorbikeCount) < 1 and len(carCount) < 1:
        control_led('green')
    elif len(motorbikeCount) == 1 and len(carCount) == 1:
        control_led('red')
    elif len(motorbikeCount) < 1 and len(carCount) == 1:
        control_led('motorbike')
    elif len(motorbikeCount) == 1 and len(carCount) < 1:
        control_led('car')

    cv2.putText(img, str(len(carCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.putText(img, str(len(motorbikeCount)), (255, 260), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    update_motorbike_count(len(motorbikeCount))
    update_car_count(len(carCount))

