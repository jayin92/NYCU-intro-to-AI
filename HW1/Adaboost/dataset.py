import os
import cv2
import numpy as np
def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    for item in os.listdir(os.path.join(dataPath, "car")):
        img = cv2.imread(os.path.join(dataPath, "car", item))
        img = cv2.resize(img, (16, 36))
        img = np.transpose(img)
        data = (img[0], 1)
        dataset.append(data)
    
    for item in os.listdir(os.path.join(dataPath, "non-car")):
        img = cv2.imread(os.path.join(dataPath, "non-car", item))
        img = cv2.resize(img, (16, 36))
        img = np.transpose(img)
        data = (img[0], 0)
        dataset.append(data)
    # End your code (Part 1)
    
    return dataset
