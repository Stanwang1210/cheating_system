import json
import cv2
import numpy as np
from utils import face_recog
import os
from MultiThread import myThread

color = json.load(open("color_table.json", "r"))  # 'RED' 'GREEN' 'BLUE' 'D_GREEN'
people_num = 3
total_pic = 6
Thread = []
for i in range(people_num):
    Thread.append(myThread(i))
    Thread[-1].start()

def face_detect(frame, faces, eyes):
    # color = json.load(open('color_table.json', 'r')) # 'RED' 'GREEN' 'BLUE' 'D_GREEN'

    Name = []
    for t in Thread:
        Name.append(t.name1)
    for t in Thread:
        Name.append(t.name2)
    print(Name)

    face_line_width = 2
    eye_line_width = 2
    detect_line_width = 5
    max_people_num = 10

    height, width, _ = frame.shape
    scale = 0.03
    y1 = int(height * scale)
    y2 = int(height * (1 - scale))
    x1 = int(width * scale)
    x2 = int(width * (1 - scale))

    expand = 50

    # Draw a rectangle around the faces
    max_people_num = min(max_people_num, faces.shape[2])
    face_count = 0
    for i in range(max_people_num):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x3, y3) = box.astype("int")
            #print(x,y,width,height)
            if x > width*2/3:
                x_pos = 2
            elif x > width/3:
                x_pos = 1
            else:
                x_pos = 0
            if y > height/2:
                y_pos = 1
            else:
                y_pos = 0
            num = 3*y_pos+x_pos
            crop = frame[y - expand : y3 + expand, x - expand : x3 + expand]

            #pic_path = 'temp/'+str(num)+'.jpg'
            pic_path = os.path.join('temp', f'{num}.jpg')
            cv2.imwrite(pic_path, crop)

            #face_recog("test.jpg")
            cv2.rectangle(frame, (x, y), (x3, y3), color["BLUE"], face_line_width)
            #cv2.line(frame, (0,height//2),(width,height//2), color["BLUE"], face_line_width)
            # cv2.imshow('t',crop)
            if num>=total_pic:
                n = 'out'
            else:
                n = Name[num] 
            if n == '':
                n = 'Unknown'
            cv2.putText(
                frame,
                n,
                ((x3), (y + y3) // 2),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=color["D_GREEN"],
                thickness=1,
            )
            for (x, y, w, h) in eyes:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color["D_GREEN"], eye_line_width
                )
            # count += 1

    cv2.putText(
                frame,
                f'{face_count} faces found',
                ((x1+50), (y2-50) ),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=color["D_GREEN"],
                thickness=1,
            )
    if face_count > 5:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color["GREEN"], detect_line_width)

    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color["RED"], detect_line_width)
    return frame, face_count
    # cv2.imshow(window_name,frame)

def shut_thread():
    for t in Thread:
        t.stop = True