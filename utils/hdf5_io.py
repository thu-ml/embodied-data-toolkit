import h5py
import numpy as np
import cv2

# Try importing torch
try:
    import torch
except ImportError:
    torch = None

def get_dataset_safe(f: h5py.File, key: str):
    """Safely retrieves a dataset or group from HDF5 file."""
    if key not in f:
        raise KeyError(f"Key '{key}' not found in HDF5 file.")
    return f[key]

def convert_to_numpy(data):
    """Converts various data types to numpy array."""
    if torch and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)
    return data

def decode_video_bytes_to_numpy(data):
    """
    Decodes a sequence of video frame bytes (e.g. from HDF5) into a numpy array (T, H, W, 3) in BGR format.
    Handles raw bytes, numpy bytes, and potential decoding errors.
    """
    frames = []
    for i, frame_bytes in enumerate(data):
        try:
            if isinstance(frame_bytes, bytes):
                # Optional: Add specific logic for double-encoded strings if needed
                pass
            
            if isinstance(frame_bytes, (bytes, np.bytes_)):
                 # Convert numpy bytes to python bytes if necessary
                 if isinstance(frame_bytes, np.bytes_):
                     frame_bytes = frame_bytes.tobytes()
            
            # Decode
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Warning: Failed to decode frame {i}")
                # Use previous frame padding
                if frames:
                    frames.append(frames[-1])
                else:
                    # If first frame fails, we skip for now, 
                    # but real implementation might need better handling (e.g. read ahead)
                    pass 
                continue
            
            # Note: cv2.imdecode returns BGR. 
            # We keep it as BGR here to be consistent with OpenCV ecosystem.
            frames.append(img)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
    
    if not frames:
        raise ValueError("No frames extracted from bytes data")
        
    return np.array(frames) # (T, H, W, 3) BGR

