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
class myThread(threading.Thread):
    def __init__(self, threadID, interval = 10):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = ''
        self.last_recog = tick()
        self.interval = interval
        self.file = os.path.join('temp', f'{self.threadID}.jpg') 
        self.stop = False
    def run(self):

        print (f"Thread {self.threadID} Start Face Recognition ..")
        while not os.path.exists(self.file) and self.stop == False:
            print(f"Waiting for Image {self.threadID} to be generated ...")
            time.sleep(2)
        #time.sleep(1)
        while  self.stop == False:#(tick() - self.last_recog) > self.interval and
            self.name = face_recog(self.file)
            self.last_recog = tick()
            while (tick() - self.last_recog) < self.interval:
                time.sleep(2)
        print('Thread', self.threadID, 'ended')
            

