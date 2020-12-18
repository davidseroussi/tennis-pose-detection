import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

cap = cv2.VideoCapture('federer.mp4')

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

    cv2.imshow('view', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    c+=1

cap.release()
cv2.destroyAllWindows()