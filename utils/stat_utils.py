import numpy as np
import torch
from pathlib import Path

def load_pt_paths(root: Path):
    """Recursively find all qpos.pt or episode_*_eef.pt files that contain action data."""
    pt_paths = []
    # We look for files named 'qpos.pt' in episode directories. 
    # Based on the pipeline structure, standardized episodes usually have 'qpos.pt'.
    for path in root.rglob("qpos.pt"):
        pt_paths.append(path)
    return sorted(pt_paths)

def min_without_outlier(data: np.ndarray, lower_percentile: float = 1.0):
    if lower_percentile <= 0.0:
        return np.min(data, axis=0)
    data = data.copy()
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    data[data <= lower_bound] = np.inf
    return np.min(data, axis=0)

def max_without_outlier(data: np.ndarray, upper_percentile: float = 99.0):
    if upper_percentile >= 100.0:
        return np.max(data, axis=0)
    data = data.copy()
    upper_bound = np.percentile(data, upper_percentile, axis=0)
    data[data >= upper_bound] = -np.inf
    return np.max(data, axis=0)

def process_file_chunk(files):
    """Process a chunk of files and return local statistics."""
    local_mins = []
    local_maxs = []
    local_sums = []
    local_sq_sums = [] # For standard deviation
    local_counts = []
    
    valid_files = 0
    
    for fn in files:
        try:
            # Load tensor
            action = torch.load(fn, map_location='cpu')
            if isinstance(action, tuple): # Handle case where load returns tuple
                action = action[0]
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            
            if action.ndim == 1:
                action = action.reshape(1, -1)
                
            if len(action) == 0:
                continue
                
            # Compute stats for this file
            mn = min_without_outlier(action, lower_percentile=0.0)
            mx = max_without_outlier(action, upper_percentile=100.0)
            s = np.sum(action, axis=0)
            sq_s = np.sum(action**2, axis=0)
            c = len(action)
            
            local_mins.append(mn)
            local_maxs.append(mx)
            local_sums.append(s)
            local_sq_sums.append(sq_s)
            local_counts.append(c)
            valid_files += 1
            
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue

    if valid_files == 0:
        return None

    return {
        "mins": local_mins,
        "maxs": local_maxs,
        "sums": local_sums,
        "sq_sums": local_sq_sums,
        "counts": local_counts
    }

