import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os.path

def get_video_frames(video_path, resize_factor=0.25):
    if not os.path.isfile(video_path):
        raise FileNotFoundError('Video path does not exist')

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:  
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))  
        frames.append(frame)
        
    cap.release()
    
    return np.stack(frames)

def get_min_distance_idx(ball_positions, wrist_keypoints):
    assert len(ball_positions) == len(wrist_keypoints)
    return np.argmin(np.linalg.norm(wrist_keypoints - ball_positions, axis=1))
    
def get_ball_positions(frames, resize_factor=1):
    ball_positions = []
    lower = (29, 77, 195)
    upper = (50, 255, 255)

    ball_positions = []

    for frame in frames:
        h, w, _ = frame.shape
        frame = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))  

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # segment yellow
        mask = cv2.inRange(hsv, lower, upper)

        # erode
        mask = cv2.erode(mask, None, iterations=2)

        # dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contours = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(contours)
            ball_positions.append([x, y])
        else:
            ball_positions.append([1e5, 1e5])
    
    return np.stack(ball_positions)

if __name__ == "__main__":
    frames = get_video_frames('videos_tennis/cut_5.mov')
    ball_positions = get_ball_positions(frames)
    ball_positions