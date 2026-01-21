import cv2
import numpy as np
from pathlib import Path

def get_video_info(video_path):
    """Get video FPS and dimensions."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    return fps, width, height

def read_video_frames(video_path):
    """Read all frames from video and return as list of RGB numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV BGR -> RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

