#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:37:28 2022

@author: wangshihang
"""

import cv2
import numpy as np
import json
from sys import exit
from time import time
from capture_win import WindowCapture
from Detector import face_detect, shut_thread
from MultiThread import myThread
import os
#color = json.load(open("color_table.json", "r"))  # 'RED' 'GREEN' 'BLUE' 'D_GREEN'
window_name = "Faces found"

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
eye_cascPath = "haarcascade_eye.xml"

"""
face_line_width = 2
eye_line_width = 2
detect_line_width = 5
"""

frame_rate = 10
frame_width, frame_height = 1280, 720
record_output_name = "output.mp4"
out = cv2.VideoWriter(
    record_output_name,
    cv2.VideoWriter_fourcc(*"mp4v"),
    frame_rate,
    (frame_width, frame_height),
    isColor=True,
)


eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascPath)

cv2.namedWindow(window_name)
# vc = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

print("load")

# initialize the WindowCapture class
path = "MLB.com | The Official Site of Major League Baseball - Google Chrome"
wincap = WindowCapture(path)

loop_time = 0



os.makedirs('temp', exist_ok=True)
"""
for i in range(people_num):
    Thread.append(myThread(i))
    Thread[-1].start()
"""
eyes = []

while True:

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = wincap.get_screenshot()
    # scale window size
    scale_window = (1280, 720)
    print(time())
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    #print(time())
    net.setInput(blob)
    #print(time())
    faces = net.forward()
    #print(time())
    #eyes = eyeCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)
    #print(time())
    face_frame, face_count = face_detect(frame, faces, eyes)
    print(f"{face_count} faces detected !!")
    #print(time())
    face_frame = cv2.resize(face_frame, scale_window)
    
    # record video
    face_frame2 = cv2.resize(face_frame, (frame_width, frame_height))
    out.write(face_frame2)
    
    cv2.imshow(window_name, face_frame)
    #print(time())
    key = cv2.waitKey(1)
    #print(time())
    # debug the loop rate
    #print("FPS {}".format(1 / (time() - loop_time)))
    #loop_time = time()
    # rval, frame = vc.read()

    if key == 27:  # exit on ESC
        break

# vc.release()
cv2.destroyWindow(window_name)
shut_thread()

