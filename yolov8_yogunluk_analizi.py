import cv2 
from ultralytics import YOLO 
import imutils 
import numpy as np
import random
from collections import defaultdict
import argparse

# Constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to YOLO model file (.onnx or .pt)")
ap.add_argument("-s", "--source", default="e.mp4", help="Path to input video file")
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
font_scale = 0.7

track_history = defaultdict(lambda: [])
kisi_sayisi = {}



while True:
    count = 0
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=1280)
    width, height, _ = frame.shape
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    cv2.rectangle(frame, (750,40), (1250,420), (255,0,0),2)
    cv2.rectangle(frame, (300,40), (700,420), (255,0,0),2)



    toplam_alan = 0
    kare_1_toplam = 0
    kare_2_toplam = 0
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        genislik = x2-x1
        yukseklik = y2-y1
        alan = genislik * yukseklik
        toplam_alan += alan
      
        class_name = results.names[int(class_id)].upper()  #car -> CAR

        track = track_history[track_id]
        track.append((cx,cy))
        if len(track) > 15:
            track.pop(0)
        
        if 750 < cx <1250 and 40 < cy <420:
            kare_1_toplam += (x2-x1) * (y2-y1)
 
        elif 400<cx<700 and 40<cy<420:
            kare_2_toplam += (x2-x1) * (y2-y1)

        points = np.hstack(track).astype("int32").reshape((-1,1,2))
        cv2.polylines(frame, [points], isClosed=False, color=(0,0,255), thickness=thickness)

        text = "ID: {} {}".format(track_id, class_name)
        cv2.putText(frame, text, (x1, y1 -5), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

        cv2.putText(frame, text, (x1, y1 -5), font, font_scale, color, thickness, cv2.LINE_AA)

    yogunluk_toplam = int((toplam_alan / (width*height))*100)
    yogunluk_alan_1 = int((kare_1_toplam / (500*380))*100)
    yogunluk_alan_2 = int((kare_2_toplam / (400*380))*100)
    
    cv2.putText(frame, "%" + str(yogunluk_toplam), (20, 40), font, 2.5, (230,34,23), 2, cv2.LINE_AA)
    cv2.putText(frame, "%" + str(yogunluk_alan_1), (300, 30), font, 1.5, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "%" + str(yogunluk_alan_2), (750, 30), font, 1.5, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()
