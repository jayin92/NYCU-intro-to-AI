import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from bonus_classify import classify


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


def detect(dataPath):
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
    cords = [] # Declare a list that stores the cordinate of each parking slots
    with open(dataPath) as file:
        num_of_parking = int(file.readline()) # Read the number total parking slots from the first line of files
        for _ in range(num_of_parking): # Iterate all lines
            tmp = file.readline() # Read a line in file
            tmp = tmp.split(" ") # Split the line using " "
            res = tuple(map(int, tmp)) # Convert the string type to int type using the built-in map function
            cords.append(res) # Append the cordinates to "cords" list
    
    cap = cv2.VideoCapture(os.path.join(dataPath, "..", "video.gif")) # Use cv2.VideoCapture to read video.gif
    frame = 0 # Counter to track current frame number
    output_gif = [] # Declare a list to store each processed frame of output.gif
    first_frame = True # A flag that will help us store the processed first frame images
    while True:
        detect_label = [] # Declare a list to store the detect results of each parking slots
        frame += 1 # Make frame number add 1
        _, img = cap.read() # Read a frame of video.gif
        if img is None: # If all frame are read, then img is None
            break # If None, then break
        for cord in cords: # Iterate all cords
            pic = crop(*cord, img) # Use * to unpack cord e.g. (x1, y1, ..., x4, y4) -> x1, y1, ..., x4, y4
            pic = cv2.resize(pic, (36, 16)) # Resize image to (36, 16)
            pic = cv2.cvtColor(pic, cv2.COLOR_RGB2HSV) # Convert image to grayscale images
            detect_label.append(classify(pic)) # Use clf.classify to detect whether the parking slot is occupied or not
                                                   # And append the result to "detect_label" list
        for i, label in enumerate(detect_label): # Iterate all detect_label
            if label: # If the model detects that this parking slot is occupied
                pos = [[cords[i][idx], cords[i][idx+1]] for idx in range(0, 8, 2)] # Add the four points of the rectangle to "pos" list
                pos[2], pos[3] = pos[3], pos[2] # swap pos[2] and pos[3]
                pos = np.array(pos, np.int32) # Convert python built-in list to numpy array
                cv2.polylines(img, [pos], color=(0, 255, 0), isClosed=True) # Use cv2.polylines to draw rectangle
    
        output_gif.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Append the result image of each frame to output_gif list (Also convert the color)
        if first_frame: # If this frame is the first frame in video.gif
            first_frame = False # After getting into this block, set first_frame to False
            cv2.imwrite(f"CV_first_frame.png", img) # Save the results of first frame
        with open(f"CV_pred.txt", "a") as txt: # Open the Adaboost_pred_<t>.txt and use append mode to add new line to file
            res = "" # Declare an empty string
            for i, label in enumerate(detect_label): # Iterate all labels
                if label:
                    res += "1" # If the parking slot is occupied, then write 1 to file
                else:
                    res += "0" # Else write 1
                if i != len(detect_label) - 1:
                    res += " " # If not the last label in a frame, then write a space to seperate each label
                else:
                    res += "\n" # Else write newline character
            txt.write(res) # Write the res string to file
    imageio.mimsave(f'results.gif', output_gif, fps=2) # Use imageio.imsave to save result gif and set fps to 2
    # End your code (Part 4)


if __name__ == "__main__":
    detect(os.path.join("data", "detect", "detectData.txt"))