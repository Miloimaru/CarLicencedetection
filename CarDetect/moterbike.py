import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
from datetime import datetime
import pytz

VIDEO_FILE = "../CarDetect/Car_Counter/Videos/test2.MOV"
MODEL_WEIGHTS = "../Yolo-Weights/yolov8l.pt"
MASK_FILE = "../CarDetect/mask3.png"
GRAPHICS_FILE = "../CarDetect/Car_Counter/graphics.png"
SAVE_FOLDER = "../CarDetect/Car_Counter/crop"
REGION_LIMITS = [400, 400, 1500, 400]
CONFIDENCE_THRESHOLD = 0.3


def initialize_yolo(model_weights):
    return YOLO(model_weights)


def read_class_names():
    return ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]


def initialize_tracker():
    return Sort(max_age=20, min_hits=3, iou_threshold=0.3)


def create_timestamped_folder():
    thailand = pytz.timezone('Asia/Bangkok')
    current_time = datetime.now(thailand)
    folder_name = current_time.strftime("%Y-%m-%d")
    folder_path = os.path.join(SAVE_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_motorbike_image(frame, x1, y1, x2, y2, folder_path):
    motorbike_img = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = f"{timestamp}.jpg"
    file_path = os.path.join(folder_path, filename)
    cv2.imwrite(file_path, motorbike_img)
    print(f"Saved {file_path}")


def process_video():
    cap = cv2.VideoCapture(VIDEO_FILE)
    yolo_model = initialize_yolo(MODEL_WEIGHTS)
    class_names = read_class_names()
    mask = cv2.imread(MASK_FILE)
    tracker = initialize_tracker()
    folder_path = create_timestamped_folder()
    totalCount = []

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRegion = cv2.bitwise_and(img, mask)
        imgGraphics = cv2.imread(GRAPHICS_FILE, cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

        results = yolo_model(imgRegion, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = class_names[cls]

                if currentClass == "motorbike" and conf > CONFIDENCE_THRESHOLD:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if not os.path.exists(SAVE_FOLDER):
                os.mkdir(SAVE_FOLDER)

            if currentClass == "motorbike" and REGION_LIMITS[0] < cx < REGION_LIMITS[2] and REGION_LIMITS[1] - 15 < cy < \
                    REGION_LIMITS[1] + 15:
                save_motorbike_image(img, x1, y1, x2, y2, folder_path)

            if REGION_LIMITS[0] < cx < REGION_LIMITS[2] and REGION_LIMITS[1] - 15 < cy < REGION_LIMITS[1] + 15:
                if id not in totalCount:
                    totalCount.append(id)

        cv2.line(img, (REGION_LIMITS[0], REGION_LIMITS[1]), (REGION_LIMITS[2], REGION_LIMITS[3]), (0, 0, 255), 5)
        cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
