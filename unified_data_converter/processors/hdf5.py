import os
import sys
import json
import numpy as np
import h5py
from typing import Dict, Any, Union

# Try importing torch
try:
    import torch
except ImportError:
    torch = None

# Try importing cv2
try:
    import cv2
except ImportError:
    cv2 = None

# Import core registry
from core.registry import ProcessorRegistry

# Import shared utils
try:
    from utils.hdf5_io import get_dataset_safe, convert_to_numpy, decode_video_bytes_to_numpy
except ImportError:
    print("Warning: Could not import utils.hdf5_io")
    pass

@ProcessorRegistry.register("extract_from_hdf5")
def process_extract_from_hdf5(target_path: str, params: dict, context: dict):
    """
    Extracts data from an HDF5 file and saves it to a specified format.
    
    Params:
        source (str): Path to source HDF5 file.
        key (str): Key/path within HDF5 file (e.g. 'observations/images/cam_high').
        format (str): Output format ('video', 'image', 'pt', 'npy', 'txt', 'json').
        index (int/slice, optional): Index to extract if dataset is an array.
        fps (int, optional): FPS for video output.
        is_video_bytes (bool, optional): If True, treats data as list of jpeg bytes.
    """
    source = params.get("source")
    key = params.get("key")
    out_format = params.get("format", "npy")
    index = params.get("index")
    
    if not source or not os.path.exists(source):
        raise FileNotFoundError(f"Source HDF5 not found: {source}")
    
    if not key:
        raise ValueError("Parameter 'key' is required for HDF5 extraction")

    with h5py.File(source, 'r') as f:
        data = get_dataset_safe(f, key)
        
        # Handle slicing/indexing
        if index is not None:
            # Simple integer index
            if isinstance(index, int):
                data = data[index]
            # String slice "start:end"
            elif isinstance(index, str) and ":" in index:
                start, end = map(lambda x: int(x) if x else None, index.split(":"))
                data = data[start:end]
            else:
                data = data[index]
        else:
            # Read full dataset
            if isinstance(data, h5py.Dataset):
                data = data[:]
            # If Group, usually we can't extract unless specific logic
            elif isinstance(data, h5py.Group):
                raise ValueError(f"Key '{key}' points to a Group, not a Dataset. Cannot extract directly.")

    # Data is now numpy array or scalar (usually)
    
    # 1. Video Output
    if out_format in ["video", "mp4", "avi"]:
        fps = params.get("fps", 30)
        is_bytes = params.get("is_video_bytes", False)
        
        # Use shared decode logic
        if is_bytes or (data.ndim == 1 and isinstance(data[0], (bytes, np.bytes_))):
            data = decode_video_bytes_to_numpy(data) # Returns (T, H, W, 3) BGR
        
        # Now data should be (T, H, W, C)
        if data.ndim != 4:
             raise ValueError(f"Data for video must be 4D (T,H,W,C), got {data.shape}")
        
        # Use cv2 to write video
        # Determine strict resolution from first frame
        H, W = data.shape[1], data.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(target_path, fourcc, fps, (W, H))
        
        for frame in data:
            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            # If data was NOT bytes, assume it might be RGB from HDF5
            # decode_video_bytes_to_numpy returns BGR, so if we didn't use it, we convert
            if not is_bytes and not (data.ndim == 1 and isinstance(data[0], (bytes, np.bytes_))): 
                 frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
        out.release()

    # 2. Image Output
    elif out_format in ["image", "jpg", "png"]:
        # Expects 2D or 3D array
        if data.ndim == 3: # HWC or CHW
            if data.shape[0] in [1, 3]: # CHW -> HWC
                data = np.transpose(data, (1, 2, 0))
        
        # Ensure uint8
        if data.dtype != np.uint8:
             data = (data * 255).astype(np.uint8) if data.max() <= 1.0 else data.astype(np.uint8)
        
        # RGB to BGR
        if data.ndim == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(target_path, data)

    # 3. PyTorch Tensor
    elif out_format in ["pt", "torch"]:
        if not torch:
            raise ImportError("Torch not installed")
        tensor = torch.from_numpy(data)
        torch.save(tensor, target_path)

    # 4. Numpy
    elif out_format in ["npy", "numpy"]:
        np.save(target_path, data)

    # 5. Text
    elif out_format == "txt":
        with open(target_path, 'w') as f:
            if isinstance(data, (np.bytes_, bytes)):
                f.write(data.decode('utf-8'))
            else:
                f.write(str(data))

    # 6. JSON
    elif out_format == "json":
        # Convert numpy types to python types
        def default_converter(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)

        with open(target_path, 'w') as f:
            json.dump(data, f, indent=4, default=default_converter)

    else:
        raise ValueError(f"Unsupported output format: {out_format}")


@ProcessorRegistry.register("create_hdf5")
def process_create_hdf5(target_path: str, params: dict, context: dict):
    """
    Creates an HDF5 file from various sources.
    
    Params:
        datasets (dict): Mapping of HDF5 path -> Source Config.
        mode (str): 'w' (overwrite), 'a' (append). Default 'w'.
    """
    datasets = params.get("datasets", {})
    mode = params.get("mode", "w")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with h5py.File(target_path, mode) as f:
        for key, config in datasets.items():
            
            # Normalize config
            if not isinstance(config, dict):
                config = {"source": config}
            
            val = None
            
            # Case A: Direct Value
            if "value" in config:
                val = convert_to_numpy(config["value"])
            
            # Case B: From Source File
            elif "source" in config:
                src = config["source"]
                if not os.path.exists(src):
                    print(f"Warning: Source for HDF5 dataset '{key}' not found: {src}")
                    continue
                    
                fmt = config.get("format")
                if not fmt:
                    # Infer format
                    ext = os.path.splitext(src)[1].lower()
                    if ext in ['.npy']: fmt = 'npy'
                    elif ext in ['.pt', '.pth']: fmt = 'pt'
                    elif ext in ['.mp4', '.avi', '.mov']: fmt = 'video'
                    elif ext in ['.json']: fmt = 'json'
                    elif ext in ['.txt']: fmt = 'txt'
                
                # Load Data
                if fmt == 'npy':
                    val = np.load(src)
                elif fmt == 'pt':
                    if not torch: raise ImportError("Torch required for .pt inputs")
                    val = torch.load(src, map_location='cpu').numpy()
                elif fmt == 'video':
                    # Read video to numpy array (T, H, W, C)
                    # Use shared util (though read_video_frames returns list, we convert to array)
                    # Note: read_video_frames returns RGB
                    try:
                        from utils.video_io import read_video_frames
                        frames = read_video_frames(src)
                        val = np.array(frames)
                    except ImportError:
                        # Fallback simple read if import fails (unlikely if structure is kept)
                        cap = cv2.VideoCapture(src)
                        frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        cap.release()
                        val = np.array(frames)

                elif fmt == 'json':
                    with open(src, 'r') as jf:
                        val = json.load(jf)
                        if isinstance(val, (dict, list)):
                            try:
                                val = np.array(val)
                                if val.dtype == np.object_: 
                                    val = json.dumps(val) 
                            except:
                                val = json.dumps(val)
                elif fmt == 'txt':
                    with open(src, 'r') as tf:
                        val = tf.read()
            
            # Write to HDF5
            if val is not None:
                # Handle String types for HDF5
                if isinstance(val, str):
                    dt = h5py.string_dtype(encoding='utf-8')
                    if key in f: del f[key]
                    f.create_dataset(key, data=val, dtype=dt)
                else:
                    if key in f: del f[key]
                    # Check compression
                    compression = config.get("compression", None)
                    if compression == "gzip" or (isinstance(val, np.ndarray) and val.size > 1000):
                         f.create_dataset(key, data=val, compression="gzip")
                    else:
                        f.create_dataset(key, data=val)

@ProcessorRegistry.register("concat_hdf5_fields")
def process_concat_hdf5_fields(target_path: str, params: dict, context: dict):
    """
    Reads multiple fields from an HDF5 file, concatenates them, and saves to target.
    """
    source = params.get("source")
    keys = params.get("keys", [])
    axis = params.get("axis", 1)
    out_format = params.get("format", "pt")
    
    if not source or not os.path.exists(source):
        raise FileNotFoundError(f"Source HDF5 not found: {source}")
    
    if not keys:
        raise ValueError("keys list is required")

    data_list = []
    
    with h5py.File(source, 'r') as f:
        for key in keys:
            data_list.append(get_dataset_safe(f, key)[:]) # Read to numpy
            
    # Concatenate
    try:
        result = np.concatenate(data_list, axis=axis)
    except Exception as e:
        raise ValueError(f"Failed to concatenate keys {keys} along axis {axis}: {e}")
        
    # Save
    if out_format in ["pt", "torch"]:
        if not torch: raise ImportError("Torch required")
        tensor = torch.from_numpy(result)
        torch.save(tensor, target_path)
    elif out_format in ["npy", "numpy"]:
        np.save(target_path, result)
    else:
        raise ValueError(f"Unsupported format: {out_format}")
