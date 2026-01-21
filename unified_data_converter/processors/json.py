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

@ProcessorRegistry.register("generate_caption")
def process_caption(target_path: str, params: dict, context: dict):
    video_path = params.get("video_path")
    use_shared = params.get("use_shared_caption", False)
    
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video for captioning not found: {video_path}")
    
    # 1. Read frames (sample 8 frames)
    # We need to read frames from the video.
    # video_utils.read_video_frames returns all frames. Might be heavy.
    # Let's use it but handle large memory carefully?
    # Or just use openCV directly here to sample?
    # video_utils has read_video_frames.
    
    frames, fps = video_utils.read_video_frames(video_path)
    total_frames = len(frames)
    
    # Sample 8 frames uniformly
    num_samples = 8
    indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)]
    sampled_frames = [frames[i] for i in indices]
    
    # 2. Generate
    task_name = context.get("task_name", "unknown task")
    caption = caption_utils.generate_caption_for_video(
        sampled_frames, 
        task_name=task_name
    )
    
    # 3. Save
    caption_utils.save_instruction_json(caption, total_frames, target_path)


@ProcessorRegistry.register("summarize_task")
def process_summarize_task(target_path: str, params: dict, context: dict):
    """
    Aggregates instructions and videos to create a task summary.
    """
    source_videos = params.get("source_videos", [])
    instructions_files = params.get("generated_instructions", [])
    
    # Example logic: Count total frames and unique instructions
    
    total_frames = 0
    unique_instructions = set()
    
    # We can iterate over instruction files
    for p in instructions_files:
        if os.path.exists(p):
            with open(p, 'r') as f:
                data = json.load(f)
                for instr in data.get("instructions", []):
                    unique_instructions.add(instr)
                    
    # We can iterate over source videos to sum duration
    # This might be slow if many files. 
    # Just an example.
    
    summary = {
        "task_name": context.get("task_name"),
        "total_episodes": len(source_videos),
        "unique_instructions": list(unique_instructions),
        "source_videos": source_videos
    }
    
    with open(target_path, 'w') as f:
        json.dump(summary, f, indent=4)


@ProcessorRegistry.register("convert_txt_to_instruction")
def convert_txt_to_instruction(target_path: str, params: dict, context: dict):
    """
    Reads a source txt file and wraps its content into the instruction JSON format.
    """
    source_path = params.get("source")
    if not source_path or not os.path.exists(source_path):
        # Fallback logic: if source is missing, use task name?
        # Or raise error. Let's raise error to be safe.
        # But if optional, maybe fallback.
        print(f"Warning: Source txt not found: {source_path}, falling back to task name.")
        instruction_text = context.get("task_name", "unknown_task").replace("_", " ")
    else:
        with open(source_path, 'r', encoding='utf-8') as f:
            instruction_text = f.read().strip()
            
    # Optional: Clean up text (e.g. remove "The whole scene is...")
    # For now, we keep it as is unless specified.
    
    data = {
       "instructions": [instruction_text],
       "sub_instructions": []
    }
    
    with open(target_path, 'w') as f:
        json.dump(data, f, indent=4)

@ProcessorRegistry.register("json_dump")
def process_json_dump(target_path: str, params: dict, context: dict):
    # Dumps the params directly, or specific content?
    # Usually the params ARE the content, or params contains 'content'
    content = params.get("content", params)
    
    with open(target_path, 'w') as f:
        json.dump(content, f, indent=4)

@ProcessorRegistry.register("extract_first_instruction_to_txt")
def extract_first_instruction_to_txt(target_path: str, params: dict, context: dict):
    """
    Extracts the first instruction from a JSON file and saves it as a single-line TXT file.
    """
    source = params.get("source")
    if not source or not os.path.exists(source):
        # Fallback to task name if source missing
        text = context.get("task_name", "unknown")
    else:
        with open(source, 'r') as f:
            data = json.load(f)
            # Try "instructions" (new format) or "seen" (old format)
            instrs = data.get("instructions", [])
            if not instrs:
                instrs = data.get("seen", [])
            
            if instrs and isinstance(instrs, list):
                text = instrs[0]
            elif isinstance(instrs, str):
                text = instrs
            else:
                text = "perform task"

    # Ensure text is clean
    text = str(text).strip()
    
    with open(target_path, 'w') as f:
        f.write(text)

@ProcessorRegistry.register("transform_robotwin_instruction")
def transform_robotwin_instruction(target_path: str, params: dict, context: dict):
    source = params.get("source")
    video_path = params.get("video_path")
    
    if not source or not os.path.exists(source):
         # If source is missing, allow fallback if task_name exists?
         # raise FileNotFoundError(f"Instruction source not found: {source}")
         # Let's use task name as fallback
         instructions = [context.get("task_name", "perform task").replace("_", " ")]
    else:     
        # Read instruction list
        with open(source, 'r') as f:
            data = json.load(f)
            # Try "seen" (original Robotwin) first, then "instructions" (converted)
            instructions = data.get("seen")
            if not instructions:
                instructions = data.get("instructions")
            
            if not instructions:
                instructions = ["perform task"]
            
            # Ensure it is a list
            if isinstance(instructions, str):
                instructions = [instructions]
            
    # Get frame count
    total_frames = 0
    if video_path and os.path.exists(video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        # If no video path, default to 0 or check params
        total_frames = params.get("total_frames", 0)

    output = {
        "instructions": instructions,
        "sub_instructions": [
            {
                "start_frame": 0,
                "end_frame": total_frames,
                "instruction": instructions # Ensure this is a list
            }
        ]
    }
    
    with open(target_path, 'w') as f:
        json.dump(output, f, indent=4)
