import json
import cv2

def face_detect(frame,faces,eyes,color):
    #color = json.load(open('color_table.json', 'r')) # 'RED' 'GREEN' 'BLUE' 'D_GREEN'

    face_line_width = 2
    eye_line_width = 2
    detect_line_width = 5

    DETECT_FACE = False
    
    height, width, _ = frame.shape
    scale = 0.03
    y1 = int(height * scale)
    y2 = int(height * (1-scale))
    x1 = int(width * scale)
    x2 = int(width * (1-scale))

    # Draw a rectangle around the faces
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
    return frame
    #cv2.imshow(window_name,frame)