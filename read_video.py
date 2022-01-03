import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os.path

cap = cv2.VideoCapture('videos_tennis/cut_7.mov')
keypoints = np.load('datasets/data_cut_7.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()['cut_7.mov']['custom'][0]

c = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = (29, 77, 195)
    upper = (50, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        contours = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(contours)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
        #cv2.circle(frame, center, 5, (0, 0, 255), -1)

        keypoints_wrist = keypoints[c][10]/4
        dist = np.linalg.norm(keypoints_wrist - np.array([x, y]))
        cv2.putText(frame, str(int(dist)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    cv2.imshow('view', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    c+=1

cap.release()
cv2.destroyAllWindows()