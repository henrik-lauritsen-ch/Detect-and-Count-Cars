#########################################################################################################
# Following ideas of Tech Watt:
# https://github.com/Tech-Watt/Python-codes/tree/main/vehicle%20detection%20and%20counting%20coures
# 
# Real Time Object Tracking With YOLOv8
#########################################################################################################


import cv2
import numpy as np
from ultralytics import YOLO
import math
from sort import *

# video_location = '/Users/henriklauritsen/documents/testtensorflow/tfodcourse/autobahnv2_A.mp4'
video_location = 'autobahnv2_A.mp4'

# define a video capture object
cap = cv2.VideoCapture(video_location)
model = YOLO('yolov8n.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines() #splitlines() makes array ['name1','name2','name3', ...]

# Counter number of cars
tracker = Sort(max_age=20)
line_out =[450, 410, 720, 420]
line_in =[800, 430, 1100, 440]
counter = []
counter_in = []

while(True):
    
    ret, frame = cap.read()
    
    if not ret:
        cap = cv2.VideoCapture(video_location)
        continue
    
    # Lines for leaving and entering the city
    cv2.line(frame,(line_out[0], line_out[1]), (line_out[2], line_out[3]), (245, 245, 245), 2)
    cv2.line(frame,(line_in[0], line_in[1]), (line_in[2], line_in[3]), (245, 245, 245), 2)
   
    detections = np.empty((0, 5))
    result = model(frame, stream=1)
    
    for info in result:
        boxes = info.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]            
            classindex = box.cls[0]
            conf = math.ceil(conf*100)
            classindex = int(classindex)
            objectDetect = classnames[classindex]
            
            if objectDetect =='car' or objectDetect == 'bus' or objectDetect == 'truck' and conf > 60:                                                      
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

                
    tracker_results = tracker.update(detections)    
    for results in tracker_results:
        x1, y1, x2, y2, id = results         
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)        
           
                       
        # Draw green box around detected car and add 'car id' + {id} on top of green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Car Id: {id}',org=[x1 + 8, y1 - 12], fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.0, color=(96, 96, 96))       
        
        
        # Calculate center of Box (any point in box would work)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + int(w/2), y1+h//2   
        
        
        # Add number of cars leaving to 'counter' and number of cars entering to 'counter_in'. We 
        # apply center point of as measure if car entered or left city
        if (line_out[0] < cx < line_out[2]) and (line_out[1] -20 < cy < line_out[3] + 20):
            cv2.line(frame,(line_out[0], line_out[1]), (line_out[2], line_out[3]), (51, 255, 51), 3)
            if counter.count(id) == 0:
                counter.append(id)

        if ((line_in[0] < cx < line_in[2]) and (line_in[1] - 20 < cy < line_in[3] + 20)):
            cv2.line(frame,(line_in[0], line_in[1]), (line_in[2], line_in[3]), (51, 51, 255), 3)
            if counter_in.count(id) == 0:
                counter_in.append(id)


    # Update text on screen with aggregated #in/#out/net change
    cv2.putText(frame, f'Number of cars leaving: {len(counter)}', org=[320, 40], fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.50, color=(255, 255, 255))

    cv2.putText(frame, f'Number of cars entering: {len(counter_in)}', org=[1020, 40], fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.50, color=(255, 255, 255))

    cv2.putText(frame, f'Net change: {len(counter_in) - len(counter)}', org=[1420, 40], fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.50, color=(255, 255, 255))

    cv2.imshow('frame', frame)  


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


