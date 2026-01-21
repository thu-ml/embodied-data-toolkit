import os
import sys
import json
import glob
import re
from typing import List, Dict

# Ensure we can import utils
# Assuming structure:
# root/
#   unified_data_converter/processors/robotics.py
#   utils/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # Up to data_process_1
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from utils import video_utils
    from utils import caption_utils
except ImportError:
    print("Warning: Could not import utils. Some processors may fail.")

from core.registry import ProcessorRegistry

@ProcessorRegistry.register("concat_video_3views")
def process_concat_video(target_path: str, params: dict, context: dict):
    cam_high = params.get("cam_high")
    cam_left = params.get("cam_left_wrist")
    cam_right = params.get("cam_right_wrist")
    resolution = params.get("resolution", [224, 224]) # Not used by util directly currently
    
    # Check if files exist
    if not (cam_high and os.path.exists(cam_high)):
        raise FileNotFoundError(f"High camera video not found: {cam_high}")
    
    # Call util
    # Note: util signature: (v_high, v_left, v_right, output_path, fps=30, resolution_config=None)
    # We ignore resolution config for now or adapt if needed
    
    video_utils.concat_videos_ffmpeg(
        cam_high, cam_left, cam_right, 
        target_path, 
        fps=30
    )