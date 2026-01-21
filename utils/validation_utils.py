import torch
import imageio
import os
import cv2
import numpy as np

def load_tensor(path):
    """Load a tensor from a pt file."""
    try:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            length = data.shape[0] if data.dim() > 0 else None
            return data, length
        if isinstance(data, (list, tuple)):
            tensor = torch.as_tensor(data)
            length = len(data)
            return tensor, length
    except Exception as e:
        print(f"Error loading tensor {path}: {e}")
    return None, None

def get_video_frame_count(path):
    """Get frame count using imageio (ffmpeg backend)."""
    try:
        with imageio.get_reader(path, "ffmpeg") as vid:
            return vid.count_frames()
    except Exception:
        # Fallback to cv2 if imageio fails or is slow
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
        except:
            return None

def is_video_black_screen(path, sample_count=10, threshold=10):
    """
    Check if a video is effectively a black screen by sampling frames.
    Returns True if video seems to be all black/static dark.
    """
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False # Cannot open, so technically not "black screen", but invalid. Handled by existence check.
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return False
            
        indices = np.linspace(0, frame_count-1, sample_count, dtype=int)
        
        is_black = True
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Check mean brightness
            if np.mean(frame) > threshold:
                is_black = False
                break
        
        cap.release()
        return is_black
    except Exception:
        return False

def check_tensor_values_range(data, abs_threshold=10):
    """
    Check if any value in tensor exceeds absolute threshold.
    Returns True if VALID (no outliers), False if INVALID.
    """
    if data is None or not torch.is_tensor(data):
        return False
    if torch.abs(data).max().item() > abs_threshold:
        return False
    return True

def check_tensor_static_zeros(data, run_length=100, dims=7):
    """
    Check if there are long runs of zeros in first/last N dimensions.
    Assumes data shape (T, D) or (T, 2*D) typically.
    Returns True if VALID, False if INVALID (has static zeros).
    """
    if not torch.is_tensor(data) or data.numel() == 0:
        return False
        
    if data.dim() == 1:
        series = data.unsqueeze(1)
    else:
        series = data.reshape(data.shape[0], -1)
        
    T, D = series.shape
    if T < run_length:
        return True # Too short to fail this check
        
    # Check first 'dims'
    if D >= dims:
        left = series[:, :dims]
        if _has_zero_run(left, run_length):
            return False
            
    # Check last 'dims' (if D is large enough, e.g. 14 for dual arm)
    # Logic: usually structure is [left_arm, right_arm]. 
    # If D=14, mid=7. series[:, 7:] is right arm.
    # We can just split by half if D is even and >= 2*dims
    
    mid = D // 2
    if D >= 2 * dims:
        # Assuming dual arm structure, check second arm
        right = series[:, mid:mid+dims]
        if _has_zero_run(right, run_length):
            return False
            
    return True

def _has_zero_run(tensor_slice, run_length):
    """Helper to find run of zeros in any dimension of shape (T, D)."""
    # Create mask of zeros
    zero_mask = (tensor_slice == 0)
    
    # We need to find if ANY column has `run_length` consecutive True
    # Can use convolution or simple loop
    for d in range(tensor_slice.shape[1]):
        col = zero_mask[:, d]
        # Find consecutive trues
        # Convert to int, diff to find edges... or just a simple counter loop for robustness
        count = 0
        max_run = 0
        for val in col:
            if val:
                count += 1
                max_run = max(max_run, count)
            else:
                count = 0
            if max_run >= run_length:
                return True
    return False
