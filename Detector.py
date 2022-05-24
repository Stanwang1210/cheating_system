import json
import cv2
import numpy as np


def scale_to_original(x,scale):
    return int(x*scale)


def face_detect(original,frame, faces, eyes, color):
    # color = json.load(open('color_table.json', 'r')) # 'RED' 'GREEN' 'BLUE' 'D_GREEN'

    face_line_width = 2
    eye_line_width = 2
    detect_line_width = 5
    max_people_num = 10

    oh,ow,_ = original.shape
    height, width, _ = frame.shape
    scale_o_h = oh/height
    scale_o_w = ow/width
    scale = 0.03
    y1 = int(oh * scale)
    y2 = int(oh * (1 - scale))
    x1 = int(ow * scale)
    x2 = int(ow * (1 - scale))

    expand = 0

    # Draw a rectangle around the faces
    max_people_num = min(max_people_num, faces.shape[2])
    face_count = 0
    for i in range(max_people_num):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x3, y3) = box.astype("int")
            x = scale_to_original(x,scale_o_w)
            y = scale_to_original(y,scale_o_h)
            x3 = scale_to_original(x3,scale_o_w)
            y3 = scale_to_original(y3,scale_o_h)
            crop = original[y-expand:y3+expand,x-expand:x3+expand]
            cv2.rectangle(original, (x, y), (x3, y3), color['BLUE'], face_line_width)
            

            #cv2.imshow('t',crop)
            cv2.putText(original, 'Test', ((x3),(y+y3)//2), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
            for (x,y,w,h) in eyes:
                x = scale_to_original(x,scale_o_w)
                y = scale_to_original(y,scale_o_h)
                w = scale_to_original(w,scale_o_w)
                h = scale_to_original(h,scale_o_h)
                cv2.rectangle(original,(x,y), (x+w,y+h),color['D_GREEN'],eye_line_width)
            #count += 1
    
    if face_count!=0:
        cv2.rectangle(original, (x1, y1), (x2, y2), color['GREEN'], detect_line_width)


    else:
        cv2.rectangle(original, (x1, y1), (x2, y2), color["RED"], detect_line_width)
    return original, face_count
    # cv2.imshow(window_name,frame)
