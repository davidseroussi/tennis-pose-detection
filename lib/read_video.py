
import cv2
video_path = '/media/david/6432-6639/DCIM/100GOPRO/GX010025.MP4'
import time

cap = cv2.VideoCapture(video_path)
resize_factor = 0.5

while True:  
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor))) 
    cv2.imshow('video', frame)
    cv2.waitKey(0)
    
cap.release()