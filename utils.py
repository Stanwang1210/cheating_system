#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 19:00:28 2022

@author: wangshihang
"""

import time
import mss
import numpy as np
import cv2
import json

face_line_width = 2
eye_line_width = 2
detect_line_width = 5
color = json.load(open('color_table.json', 'r')) # 'RED' 'GREEN' 'BLUE' 'D_GREEN'
window_name = "Faces found"
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
eye_cascPath = 'haarcascade_eye.xml'


def tick():
    return time.time()

def record_screen(length=10, frame_height=720, frame_width=1280, frame_rate = 10.0, record_output_name = "output.mp4"):
    
    sct = mss.mss()
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))
    out = cv2.VideoWriter(record_output_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height), isColor=True)
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascPath)
    
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    start = tick()
    end = tick()
    taken_time = 0
    while (end - start - taken_time < length) :
        print(f'length is {end - start - taken_time} secs' )
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inner_start = tick()
        frame = detect_face(frame, net, eyeCascade)
        taken_time += (tick() - inner_start)
    # cv2.putText(
    #     frame,
    #     "FPS: %f" % (1.0 / (time.time() - last_time)),
    #     (10, 10),s
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     2,
    # )q
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_width, frame_height))
        out.write(frame)
        end = tick()
        
    print(f'Taken time is {end -start} secs' )
def detect_face(frame, net, eyeCascade):
    global face_line_width, eye_line_width, detect_line_width, color 
    height, width, _ = frame.shape
    print('Frame height is ', height)
    print('Frame width is ', width)
    scale = 0.03
    y1 = int(height * scale)
    y2 = int(height * (1-scale))
    x1 = int(width * scale)
    x2 = int(width * (1-scale))
    face_count = 0
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    eyes = eyeCascade.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 4)
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # print('index ', i)
            face_count += 1
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x3, y3) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x3, y3), color['BLUE'], face_line_width)
            for (x,y,w,h) in eyes:
                cv2.rectangle(frame,(x,y), (x+w,y+h),color['D_GREEN'],eye_line_width)
    print(f'{face_count} faces detect !! ')
    if face_count > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color['GREEN'], detect_line_width)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color['RED'], detect_line_width)
    return frame
record_screen(3)