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

# Capture screen path
path = "MLB.com | The Official Site of Major League Baseball - Google Chrome"
wincap = WindowCapture(path)

# Window name
window_name = "Faces found"

# Models
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
eye_cascPath = "haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascPath)

# record video
rec = False
if rec:
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

# Setup
cv2.namedWindow(window_name)
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

print("Loading...")

loop_time = 0
eyes = []

while True:
    frame = wincap.get_screenshot()
    # scale window size
    scale_window = (1280, 720)

    # Face Detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    net.setInput(blob)
    faces = net.forward()
    #eyes = eyeCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)
    face_frame, face_count = face_detect(frame, faces, eyes)
    print(f"{face_count} faces detected !!")
    face_frame = cv2.resize(face_frame, scale_window)
    
    # record video
    if rec:
        face_frame2 = cv2.resize(face_frame, (frame_width, frame_height))
        out.write(face_frame2)
    
    cv2.imshow(window_name, face_frame)

    key = cv2.waitKey(1)

    # debug the loop rate
    #print("FPS {}".format(1 / (time() - loop_time)))
    #loop_time = time()

    if key == 27:  # exit on ESC
        break

cv2.destroyWindow(window_name)
shut_thread()

