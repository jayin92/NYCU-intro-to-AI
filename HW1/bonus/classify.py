import cv2
import numpy as np

def classify(img):
    v = img[:,:,2]
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    if 40 <= np.argmax(hist_v) and np.argmax(hist_v) <= 200 and 27 <= np.max(hist_v):
        return False
    else:
        return True
