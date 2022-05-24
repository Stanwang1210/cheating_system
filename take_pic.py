#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:51:41 2022

@author: wangshihang

"""

import cv2
import time

cap = cv2.VideoCapture(0)


while cap.isOpened():
    time.sleep(1)
    success, img = cap.read()
    if success:
        cv2.imwrite("wang.jpg", img)
        cap.release() 
        break