import cv2
from ultralytics import YOLO
import math
import torch
import numpy as np
import datetime 
import os
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200)
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
RED = (0, 0, 255)
video_cap = cv2.VideoCapture(0)
output_width = 640
output_height = 480
output_directory = "output"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
current_datetime = datetime.datetime.now()
output_file_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + ".avi"

output_path = os.path.join(output_directory, output_file_name)
out = cv2.VideoWriter(output_path, fourcc, 30.0, (output_width, output_height))
CLASS = ["green", "red"]
model = YOLO("modelname.pt")


ct = 0
while video_cap.isOpened():
    max_red_area = 0
    max_green_area = 0
    bgreen_tensor = None
    bred_tensor = None
    ret, frame = video_cap.read()
    
    if ret:
        frame = cv2.resize(frame, (output_width, output_height))
        
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=(output_width, output_height))

        for box in results[0].boxes:
            w = box.xywh[0][2]
            h = box.xywh[0][3]
            area = max(w * w, h * h)

            if CLASS[int(box.cls)] == "green":
                if max_green_area < area:
                    max_green_area = area
                    bgreen_tensor = box
            elif CLASS[int(box.cls)] == "red":
                if max_red_area < area:
                    max_red_area = area
                    bred_tensor = box


        if(bgreen_tensor!=None):
            x1_green,y1_green,x2_green,y2_green=bgreen_tensor.xyxy[0]
            x1_green,y1_green,x2_green,y2_green = map(int, [x1_green, y1_green, x2_green, y2_green])
            cv2.rectangle(frame, (x1_green, y1_green), (x2_green, y2_green), GREEN, 2)
            
        if(bred_tensor!=None):
            x1_red,y1_red,x2_red,y2_red = bred_tensor.xyxy[0]
            x1_red,y1_red,x2_red,y2_red = map(int, [x1_red,y1_red,x2_red,y2_red])
            cv2.rectangle(frame, (x1_red, y1_red), (x2_red, y2_red), RED, 2)

        if bgreen_tensor and bred_tensor:
            bottom_x_y = (round(640/2), 480)
            bottom_x_y_kiri = (0, 480)
            bottom_x_y_kanan = (640, 480)
            cv2.line(frame, (x2_green, y1_green), (x1_red, y1_red), color=(225, 0, 0), thickness=2)
            line_x_top_center = (x2_green+x1_red) // 2 
            line_y_top_center = (y1_green+y1_red) // 2
            line_center_x_y = (line_x_top_center, line_y_top_center)
            cv2.line(frame, bottom_x_y, line_center_x_y, thickness=2, color=(255,254,254))

            rudder_angle = ((- bottom_x_y[0] + line_x_top_center ) // 5) + 70
            
            if(rudder_angle > 180):
                rudder_angle = 0
            
            if(rudder_angle < 0):
                rudder_angle = 180
        elif bgreen_tensor:
            if bgreen_tensor.xyxy[0][0]>170:
                rudder_angle=90
            else:
                rudder_angle=0
        
        elif bred_tensor:
            if bred_tensor.xyxy[0][0]<500:
                rudder_angle=180
            else:
                rudder_angle=90

        else:     
            rudder_angle = 90 
        cv2.putText(frame, str(rudder_angle), org=(20, 20), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=RED)
        speed = 100
        command = f"{rudder_angle},{speed}\n".encode()
        ser.write(command)
        lines=ser.readline()

        out.write(frame)
        cv2.imshow("YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_cap.release()
out.release()
cv2.destroyAllWindows()
