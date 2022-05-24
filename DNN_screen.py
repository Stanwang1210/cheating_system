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
from Detector import face_detect

color = json.load(open("color_table.json", "r"))  # 'RED' 'GREEN' 'BLUE' 'D_GREEN'
window_name = "Faces found"
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
eye_cascPath = "haarcascade_eye.xml"
face_line_width = 2
eye_line_width = 2
detect_line_width = 5
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
"""
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
"""
path = "MLB.com | The Official Site of Major League Baseball - Google Chrome"

# initialize the WindowCapture class
wincap = WindowCapture(path)


loop_time = 0
while True:
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = wincap.get_screenshot()
    # scale window size

    scale_window = (1280,720)
    original = frame
    

    """
    DETECT_FACE = False
    
    height, width, _ = frame.shape
    scale = 0.03
    y1 = int(height * scale)
    y2 = int(height * (1-scale))
    x1 = int(width * scale)
    x2 = int(width * (1-scale))
    #print(x1,y1,x2,y2)
    """

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    net.setInput(blob)
    faces = net.forward()

    eyes = eyeCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)
    face_frame, face_count = face_detect(original,frame, faces, eyes, color)
    print(f"{face_count} faces detected !!")
    face_frame = cv2.resize(face_frame, scale_window)

    """
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
    #     roi_gray = gray[y:y+w, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    # eyes = eyeCascade.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 4)
    # print("Found {0} faces!".format(len(faces)))
    # print('face count ', faces.shape[0])
    # if faces.shape[2] == 0:
    # else:

    # # Draw a rectangle around the faces
    count = 0
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            DETECT_FACE = True
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x3, y3) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x3, y3), color['BLUE'], face_line_width)
            for (x,y,w,h) in eyes:
                cv2.rectangle(frame,(x,y), (x+w,y+h),color['D_GREEN'],eye_line_width)
            count += 1
    if DETECT_FACE:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color['GREEN'], detect_line_width)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color['RED'], detect_line_width)
    """
    #face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (frame_width, frame_height))
    out.write(face_frame)
    cv2.imshow(window_name, face_frame)

    key = cv2.waitKey(1)

    # debug the loop rate
    print("FPS {}".format(1 / (time() - loop_time)))
    loop_time = time()
    # rval, frame = vc.read()

    if key == 27:  # exit on ESC
        break

# vc.release()
cv2.destroyWindow(window_name)
