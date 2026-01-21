import torch
import numpy as np

def get_trim_indices(qpos_tensor: torch.Tensor, threshold: float):
    """
    Analyze qpos tensor to find start and end indices of movement.
    
    Args:
        qpos_tensor: shape (T, D)
        threshold: L2 norm threshold for movement detection
        
    Returns:
        (trim_start, trim_end) or (None, None)
        trim_start: inclusive
        trim_end: exclusive (for python slice)
    """
    if qpos_tensor.ndim != 2:
        return None, None
    
    # Calculate difference between consecutive frames
    diffs = qpos_tensor[1:] - qpos_tensor[:-1]
    # L2 norm
    norms = torch.linalg.norm(diffs.float(), dim=1)
    
    # Find indices where movement > threshold
    moved_indices = torch.where(norms > threshold)[0]
    
    if len(moved_indices) == 0:
        return None, None 
        
    # Logic from trim_dataset.py:
    # First move at idx means change happened between idx and idx+1. 
    # So we keep from idx+1.
    trim_start = moved_indices[0].item() + 1
    
    # Last move at idx. We keep until idx+1. 
    # Python slice end is exclusive, so we use idx+2.
    trim_end = moved_indices[-1].item() + 2
    
    return trim_start, trim_end

def load_tensor(path):
    return torch.load(path)

def save_tensor(tensor, path):
    torch.save(tensor, path)
