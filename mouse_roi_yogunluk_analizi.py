import cv2 
import numpy as np
import random
from collections import defaultdict
from ultralytics import YOLO 
import imutils 
import argparse

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            rect_width = param["width"]
            rect_height = param["height"]

            x1 = max(0, x - rect_width // 2)
            y1 = max(0, y - rect_height // 2)
            x2 = min(frame.shape[1], x + rect_width // 2)
            y2 = min(frame.shape[0], y + rect_height // 2)

            img_copy = frame.copy()
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow("frame", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_width = param["width"]
        rect_height = param["height"]

        x1 = max(0, x - rect_width // 2)
        y1 = max(0, y - rect_height // 2)
        x2 = min(frame.shape[1], x + rect_width // 2)
        y2 = min(frame.shape[0], y + rect_height // 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("frame", frame)
        param["rect_coords"] = [x1, y1, x2, y2]
        param["count"] = 0  # Add initial count of people within the rectangle

# Constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to YOLO model file (.onnx or .pt)")
ap.add_argument("-s", "--source", default="e.mp4", help="Path to input video file")
ap.add_argument("-w", "--width", type=int, default=300, help="Width of the rectangle")
ap.add_argument("-H", "--height", type=int, default=300, help="Height of the rectangle")
args = vars(ap.parse_args())

# Loading YOLO model
model = YOLO(args["model"])

# Opening video capture
if args["source"] == "0":
    video_path = 0  # Webcam
else:
    video_path = args["source"]

cap = cv2.VideoCapture(video_path)

thickness = 1
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 1.5

track_history = defaultdict(lambda: [])

drawing = False  # true if mouse is pressed
ix, iy = -1, -1

cv2.namedWindow("frame")

# Store rectangle dimensions in a dictionary
rect_params = {"width": args["width"], "height": args["height"]}

cv2.setMouseCallback("frame", draw_rectangle, rect_params)

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=1280)
    width, height, _ = frame.shape
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")

    if rect_params.get("rect_coords"):
        x1, y1, x2, y2 = rect_params["rect_coords"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    toplam_alan = 0
    count_within_region = 0

    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if rect_params.get("rect_coords") and rect_params["rect_coords"][0]  < cx < rect_params["rect_coords"][2] and rect_params["rect_coords"][1]  < cy < rect_params["rect_coords"][3]:
            genislik = x2 - x1
            yukseklik = y2 - y1
            alan = args["width"] * args["height"]
            toplam_alan += alan
            count_within_region += 1

    cv2.putText(frame, "Bolgedeki Kisi Sayisi: " + str(count_within_region), (20, 40), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    yogunluk_toplam = int((toplam_alan / (width * height)) * 100)

    cv2.putText(frame, "Bolgenin Yogunlugu %" + str(yogunluk_toplam), (20, 20), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
