#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:17:20 2022

@author: wangshihang
"""

import threading
import time
import os
from utils import face_recog, tick
import cv2
class myThread(threading.Thread):
    def __init__(self, threadID, interval = 5,t_th = 3, t_p = 6):
        threading.Thread.__init__(self)
        
        self.threadID = threadID
        self.name1 = ''
        self.name2 = ''
        self.last_recog = tick()
        self.interval = interval
        self.file = os.path.join('temp', f'{self.threadID}.jpg') 
        self.stop = False
        self.total_thread = t_th
        self.total_pic = t_p
        self.add = t_p//t_th
        
    def run(self):
        
        num = self.threadID
        init = 0
        print (f"Thread {self.threadID} Start Face Recognition ..")
        while(self.stop == False):
            
            
            if not os.path.exists(self.file) :
                print(f"Waiting for Image {num} to be generated ...")
                time.sleep((self.threadID+1)*0.5)
            #time.sleep(1)
            else:#(tick() - self.last_recog) > self.interval and
                if init == 0:

                    try:
                        self.name1 = face_recog(self.file)
                    except:
                        self.name1 = "Unknown"#"Can't detect face"
                else:

                    try:
                        self.name2 = face_recog(self.file)
                    except:
                        self.name2 = "Unknown"#"Can't detect face"
                #cv2.namedWindow('test')
                #cv2.namedWindow('test')
                #cv2.imshow('test',self.file)
                self.last_recog = tick()
                while (tick() - self.last_recog) < self.interval:
                    time.sleep(3)
            num = (num + self.total_thread)%self.total_pic
            self.file = os.path.join('temp', f'{num}.jpg') 
            init = (init +1) % self.add
        print('Thread', self.threadID, 'ended')
            

