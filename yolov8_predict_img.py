import cv2 
from ultralytics import YOLO 
import imutils 
import numpy as np
import random

img_path = ""
model_path = ""
img = cv2.imread(img_path)
model = YOLO(model_path)

thickness = 2
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 0.7


img = imutils.resize(img, width=600)
color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
results = model.track(img, persist=True, verbose = False)[0]
#track_ids = results.boxes.id.int().cpu().tolist()
bboxes = np.array(results.boxes.data.tolist(), dtype = "int")

for box in bboxes:
        
    x1, y1, x2, y2, track_id, score, class_id = box
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
        
    class_name = results.names[int(class_id)].upper()  

       

    text = "ID: {} {}".format(track_id, class_name)
    cv2.putText(img, text, (x1, y1 -5), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    cv2.imshow("img",img)


cv2.waitKey(0)
cv2.destroyAllWindows()