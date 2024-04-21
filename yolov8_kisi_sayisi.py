import cv2 
from ultralytics import YOLO 
import imutils 
import numpy as np
import random
from collections import defaultdict

cap = cv2.VideoCapture("e.mp4")
model = YOLO("best.onnx")

thickness = 1
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 0.7

track_history = defaultdict(lambda: [])


while True:
    count = 0
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=1280)
    color = (255, 0, 0)
    results = model.track(frame, persist=True, verbose = False)[0]
    #track_ids = results.boxes.id.int().cpu().tolist()
    bboxes = np.array(results.boxes.data.tolist(), dtype = "int")
    cv2.line(frame, (0,450),(1280,450),(0,255,0), 2)

    for box in bboxes:
        count += 1
        
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
      
        class_name = results.names[int(class_id)].upper()  #car -> CAR

            

        track = track_history[track_id]
        track.append((cx,cy))
        if len(track)>15:
            track.pop(0)



        points = np.hstack(track).astype("int32").reshape((-1,1,2))
        cv2.polylines(frame, [points], isClosed=False, color = (0,0,255), thickness = thickness)

        text = "ID: {} {}".format(track_id, class_name)
        cv2.putText(frame, text, (x1, y1 -5), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

    kisi_sayisi = "Kisi Sayisi: {} ".format(count)
    cv2.putText(frame, kisi_sayisi, (20,30), font, 1.5, color, thickness, cv2.LINE_AA)


    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()