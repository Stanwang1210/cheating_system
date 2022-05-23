import cv2 
import numpy as np
import os
from time import time
from capture_win import WindowCapture

path = 'MLB.com | The Official Site of Major League Baseball - Google Chrome'

# initialize the WindowCapture class
wincap = WindowCapture(path)

loop_time = time()
while(True):

    # get an updated image
    screenshot = wincap.get_screenshot()
    # scale window size
    scale_window = (1200,675)
    fr = cv2.resize(screenshot, scale_window)
    cv2.imshow('Computer Vision', fr)

    # debug the loop rate
    #print('FPS {}'.format(1 / (time() - loop_time)))
    #loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print('Done.')



