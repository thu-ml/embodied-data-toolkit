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
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return False
            
        indices = np.linspace(0, frame_count-1, sample_count, dtype=int)
        
        is_black = True
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if np.mean(frame) > threshold:
                is_black = False
                break
        
        cap.release()
        return is_black
    except Exception:
        return False

def is_video_corrupted(path):
    """
    Check if a video file is corrupted or cannot be opened.
    Tries to open and read a few frames. Returns True if corrupted.
    """
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return True
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return True
        # Try reading first, middle, and last frame
        test_indices = [0, frame_count // 2, max(0, frame_count - 1)]
        read_ok = 0
        for idx in test_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                read_ok += 1
        cap.release()
        return read_ok == 0
    except Exception:
        return True

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
        return True
        
    if D >= dims:
        left = series[:, :dims]
        if _has_zero_run(left, run_length):
            return False
            
    mid = D // 2
    if D >= 2 * dims:
        right = series[:, mid:mid+dims]
        if _has_zero_run(right, run_length):
            return False
            
    return True

def _has_zero_run(tensor_slice, run_length):
    """Helper to find run of zeros in any dimension of shape (T, D)."""
    zero_mask = (tensor_slice == 0)
    for d in range(tensor_slice.shape[1]):
        col = zero_mask[:, d]
        count = 0
        for val in col:
            if val:
                count += 1
                if count >= run_length:
                    return True
            else:
                count = 0
    return False


def _has_repeat_run(block, run_len):
    """
    Check if a (T, dims) block has run_len+ consecutive identical rows.
    Treats all dims as a single vector: row t == row t-1 means repeat.
    """
    diffs = (block[1:] - block[:-1]).abs().sum(dim=1)
    count = 0
    for d in diffs:
        if d.item() == 0.0:
            count += 1
            if count >= run_len:
                return True
        else:
            count = 0
    return False


def check_tensor_static_repeat(data, run_length=100, dims=7):
    """
    Check if first/last N dims (as a group) have >run_length consecutive
    identical rows. For dual-arm (D>=14), checks both arm groups.
    Returns True if VALID, False if INVALID.
    """
    if not torch.is_tensor(data) or data.numel() == 0:
        return False

    if data.dim() == 1:
        series = data.unsqueeze(1)
    else:
        series = data.reshape(data.shape[0], -1).float()

    T, D = series.shape
    if T <= run_length:
        return True

    if D >= dims and _has_repeat_run(series[:, :dims], run_length):
        return False

    mid = D // 2
    if D >= 2 * dims and _has_repeat_run(series[:, mid:mid + dims], run_length):
        return False

    return True


def check_tensor_group_all_zeros(data, dims=7):
    """
    Check if first/last N dims are entirely zero across ALL timesteps.
    For dual-arm (D>=14), checks both arm groups independently.
    Returns True if VALID, False if INVALID (all zeros detected).
    """
    if not torch.is_tensor(data) or data.numel() == 0:
        return False

    if data.dim() == 1:
        series = data.unsqueeze(1)
    else:
        series = data.reshape(data.shape[0], -1)

    T, D = series.shape

    if D >= dims and (series[:, :dims] == 0).all().item():
        return False

    mid = D // 2
    if D >= 2 * dims and (series[:, mid:mid + dims] == 0).all().item():
        return False

    return True
