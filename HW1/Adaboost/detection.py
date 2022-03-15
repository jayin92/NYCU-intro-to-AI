import os
import re
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
import imageio
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf, t=10):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    cords = []
    with open(dataPath) as file:
        num_of_parking = int(file.readline())
        for _ in range(num_of_parking):
            tmp = file.readline()
            tmp = tmp.split(" ")
            res = tuple(map(int, tmp))
            cords.append(res)
    cap = cv2.VideoCapture(os.path.join(dataPath, "..", "video.gif"))
    frame = 0
    output_gif = []
    first_frame = True
    while True:
        detect_label = []
        frame += 1
        _, img = cap.read()
        if img is None:
            break
        for cord in cords:
            pic = crop(*cord, img)
            pic = cv2.resize(pic, (36, 16))
            pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
            detect_label.append(clf.classify(pic))
        for i, label in enumerate(detect_label):
            if label:
                pos = [[cords[i][idx], cords[i][idx+1]] for idx in range(0, 8, 2)]
                pos[2], pos[3] = pos[3], pos[2]
                pos = np.array(pos, np.int32)
                cv2.polylines(img, [pos], color=(0, 255, 0), isClosed=True)
    
        output_gif.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if first_frame:
            first_frame = False
            cv2.imwrite(f"Adaboost_first_frame_{t}.png", img)
        with open(f"Adaboost_pred_{t}.txt", "a") as txt:
            res = ""
            for i, label in enumerate(detect_label):
                if label:
                    res += "1"
                else:
                    res += "0"
                if i != len(detect_label) - 1:
                    res += " "
                else:
                    res += "\n"
            txt.write(res)
    imageio.mimsave(f'results_{t}.gif', output_gif, fps=2)
    # End your code (Part 4)
