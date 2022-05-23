#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:48:20 2022

@author: wangshihang
"""

import cv2

window_name = "Faces found"
cv2.namedWindow(window_name)
vc = cv2.VideoCapture(0)

# face_cascPath = "haarcascade_frontalface_alt.xml"
face_cascPath = "haarcascade_frontalface_default.xml"
eye_cascPath = 'haarcascade_eye.xml'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascPath)
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascPath)
print('load')
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    height, width, _ = frame.shape
    scale = 0.03
    y1 = int(height * scale)
    y2 = int(height * (1-scale))
    x1 = int(width * scale)
    x2 = int(width * (1-scale))
    #print(x1,y1,x2,y2)

    #print(height,width)
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
    #     roi_gray = gray[y:y+w, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     eyes = eyeCascade.detectMultiScale(roi_gray, 1.3, 5)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
    eyes = eyeCascade.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 4)
    # print("Found {0} faces!".format(len(faces)))
    if len(faces) == 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #print(x,y,x+w,y+h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if x>x1 and y>y1 and x+w<x2 and y+h<y2:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 139, 34), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0, 255, 0),5)
    cv2.imshow(window_name,frame)
    key = cv2.waitKey(20)
    rval, frame = vc.read()
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow(window_name)
