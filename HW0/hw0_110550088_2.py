import cv2
import numpy as np

cap = cv2.VideoCapture("data/video.mp4")

for i in range(10):
    _, frame1 = cap.read()
    _, frame2 = cap.read()

cap.release()

diff = cv2.absdiff(frame1, frame2)
diff[:,:,0] = np.zeros([diff.shape[0], diff.shape[1]])
diff[:,:,2] = np.zeros([diff.shape[0], diff.shape[1]])
res = np.hstack([frame1, diff])


cv2.imwrite("hw0_110550088_2.png", res)