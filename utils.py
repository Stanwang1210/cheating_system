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

# from deepface import DeepFace
import glob
from sys import exit
import face_recognition

face_line_width = 2
eye_line_width = 2
detect_line_width = 5
maximum_face_number = 10
color = json.load(open("color_table.json", "r"))  # 'RED' 'GREEN' 'BLUE' 'D_GREEN'
window_name = "Faces found"
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
eye_cascPath = "haarcascade_eye.xml"
pic_db = glob.glob("./data_base/*.jpg")
data_bace_encodings = []


def tick():
    return time.time()


def initialize(reload=False):
    global data_bace_encodings, pic_db
    if reload:
        print("Reload Data Base ..")
        for pic in pic_db:
            img = face_recognition.load_image_file(pic)
            try:
                data_bace_encodings.append(face_recognition.face_encodings(img)[0])
            except:
                print("Can not dicriminate ", pic)

        np.save("data_base_encodings.npy", data_bace_encodings)
    else:
        data_bace_encodings = np.load("data_base_encodings.npy")


def face_recog(img_path):
    global data_bace_encodings

    img = face_recognition.load_image_file(img_path)
    print(img.shape)
    
    try:
        unknown_encoding = face_recognition.face_encodings(img)[0]
    except:
        print("Can not detect face")
        return "Unknown"#"Can not detect face"
    results = face_recognition.compare_faces(
        data_bace_encodings, unknown_encoding, tolerance=0.4
    )
    #print(results)
    name_1 = img_path.replace(".jpg", "").split("\\")[-1]
    name = ""
    for i in range(len(results)):
        if results[i]:
            name_2 = pic_db[i].replace(".jpg", "").split("\\")[-1]
            name += name_2
            name += " "

            print(f"{name_1} is {name_2}")
    print(name)
    return name


def entity_checking(img):
    global pic_db

    # print(pic_db)
    # exit()
    model = ["Facenet", "VGG-Face", "Facenet512"]
    img1 = [[img, img2] for img2 in pic_db]
    result = DeepFace.verify(
        img1_path=img1,
        model_name=model[2],
        enforce_detection=False,
        distance_metric="euclidean_l2",
    )
    print("")
    # result = DeepFace.find(
    #     img_path=img,
    #     db_path="data_base",
    #     model_name="Facenet",
    #     distance_metric="euclidean_l2",
    #     enforce_detection=False,
    # )
    # print(result)
    # exit(0)
    max_idx = 0
    min_dis = 100
    for i in range(len(img1)):
        pair = f"pair_{i+1}"
        if result[pair]["distance"] < min_dis:
            max_idx = i
            min_dis = result[pair]["distance"]

        print(
            f"{img1[i][0]} and  {img1[i][1]} are same person: {result[pair]['verified']}"
        )
        print(f"distance is {result[pair]['distance']}")
        print("------------------------------------------")
    name_1 = img.replace("jpg", "").split("/")[-1]
    name_2 = pic_db[max_idx].replace(".jpg", "").split("/")[-1]
    print(f"The most similar one of {name_1} is  {name_2}")
    return name_2


def record_screen(
    length=10,
    frame_height=720,
    frame_width=1280,
    frame_rate=20.0,
    record_output_name="output.mp4",
):

    sct = mss.mss()
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))
    out = cv2.VideoWriter(
        record_output_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (frame_width, frame_height),
        isColor=True,
    )
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascPath)

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    start = tick()
    end = tick()
    taken_time = 0
    Frame = []
    while end - start - taken_time < length:
        print(f"length is {end - start - taken_time} secs")
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inner_start = tick()
        frame = detect_face(frame, net, eyeCascade)
        print("detect_time", tick() - inner_start)
        taken_time += tick() - inner_start
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
        Frame.append(frame)
        end = tick()
    print(f"Taken time is {end -start} secs")
    for frame in Frame:
        out.write(frame)


def detect_face(frame, net, eyeCascade):
    global face_line_width, eye_line_width, detect_line_width, color, maximum_face_number
    height, width, _ = frame.shape
    print("Frame height is ", height)
    print("Frame width is ", width)
    scale = 0.03
    y1 = int(height * scale)
    y2 = int(height * (1 - scale))
    x1 = int(width * scale)
    x2 = int(width * (1 - scale))
    face_count = 0
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    net.setInput(blob)
    faces = net.forward()
    eyes = eyeCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4)
    maximum_face_number = min(frame.shape[2], maximum_face_number)
    for i in range(maximum_face_number):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # print('index ', i)
            face_count += 1
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x3, y3) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x3, y3), color["BLUE"], face_line_width)
            for (x, y, w, h) in eyes:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color["D_GREEN"], eye_line_width
                )
    print(f"{face_count} faces detect !! ")
    if face_count > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color["GREEN"], detect_line_width)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color["RED"], detect_line_width)
    return frame


# record_screen(3)
print("Start Initialize")
initialize()
"""
print('Finish Initialize')
pic = glob.glob('./*.jpg')
for p in pic:
    print('---------------------')
    print(p)
    face_recog(p)
"""
1
