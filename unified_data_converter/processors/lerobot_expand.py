import os
import json
import torch
import numpy as np
import shutil
from pathlib import Path
from multiprocessing import Pool
from core.registry import ProcessorRegistry

# Import shared utils
try:
    from utils.parquet_io import read_parquet_to_tensor
except ImportError:
    # Try adjusting path if running from unified_data_converter root
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.parquet_io import read_parquet_to_tensor
except ImportError:
    print("Warning: Could not import utils.parquet_io")
    pass

@ProcessorRegistry.register("expand_lerobot_episodes")
def process_expand_lerobot_episodes(target_path: str, params: dict, context: dict):
    """
    Expands LeRobot episodes into standard format.
    Supports parallel execution if 'workers' is specified in params.
    """
    target_root = Path(target_path)
    
    meta_dir = Path(params.get("meta_dir"))
    source_task_root = meta_dir.parent
    
    workers = params.get("workers", 1)
    
    videos_dir = Path(params.get("videos_dir"))
    data_dir = Path(params.get("data_dir"))
    
    camera_map = params.get("cameras", {})
    main_video_key = params.get("main_video_key", "observation.images.image")
    action_key = params.get("action_key", "action")
    save_action_as = params.get("save_action_as", "qpos")
    
    if not meta_dir.exists():
        raise FileNotFoundError(f"Meta dir not found: {meta_dir}")

    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    
    if not episodes_path.exists():
        raise FileNotFoundError("episodes.jsonl missing")
        
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
            
    task_map = {}
    if tasks_path.exists():
        with open(tasks_path, 'r') as f:
            for line in f:
                t = json.loads(line)
                task_map[t["task_index"]] = t["task"]
                
    print(f"Processing task {source_task_root.name}: {len(episodes)} episodes with {workers} workers")
    
    func_args = []
    for ep in episodes:
        func_args.append({
            "ep": ep,
            "task_map": task_map,
            "data_dir": str(data_dir),
            "videos_dir": str(videos_dir),
            "target_root": str(target_root),
            "camera_map": camera_map,
            "main_video_key": main_video_key,
            "action_key": action_key,
            "save_action_as": save_action_as
        })
        
    if workers > 1:
        with Pool(workers) as p:
            p.map(_process_single_episode_wrapper, func_args)
    else:
        for arg in func_args:
            _process_single_episode_wrapper(arg)

def _process_single_episode_wrapper(args):
    """Wrapper to unpack args and run logic."""
    try:
        _process_single_episode(**args)
    except Exception as e:
        print(f"Error processing episode: {e}")

def _process_single_episode(ep, task_map, data_dir, videos_dir, target_root, camera_map, main_video_key, action_key, save_action_as):
    ep_idx = ep["episode_index"]
    task_idx = ep.get("task_index", 0)
    length = ep.get("length")
    
    data_dir = Path(data_dir)
    videos_dir = Path(videos_dir)
    target_root = Path(target_root)
    
    ep_filename = f"episode_{ep_idx:06d}"
    
    # Search parquet
    found = list(data_dir.rglob(f"{ep_filename}.parquet"))
    if not found:
        return
    
    parquet_path = found[0]
    chunk_name = parquet_path.parent.name
    video_chunk_dir = videos_dir / chunk_name
    
    # Target
    dest_ep_dir = target_root / f"episode_{ep_idx}"
    dest_raw_video_dir = dest_ep_dir / "raw_video"
    dest_ep_dir.mkdir(parents=True, exist_ok=True)
    dest_raw_video_dir.mkdir(exist_ok=True)
    
    # Instructions
    task_desc = task_map.get(task_idx, "")
    instr_data = {
        "instructions": [task_desc] if task_desc else [],
        "sub_instructions": [
            {
                "start_frame": 0,
                "end_frame": length,
                "instruction": [task_desc] if task_desc else []
            }
        ],
        "total_frames": length
    }
    with open(dest_ep_dir / "instructions.json", 'w') as f:
        json.dump(instr_data, f, indent=4, ensure_ascii=False)
        
    # Actions (Using reusable utils)
    try:
        actions = read_parquet_to_tensor(parquet_path, column_key=action_key, fuzzy_search_key="action")
        if actions is not None:
            torch.save(actions, dest_ep_dir / f"{save_action_as}.pt")
        else:
            print(f"Warning: Action column '{action_key}' not found in {parquet_path}")
    except ImportError:
        print("Skipping parquet read: pyarrow not installed")
    except Exception as e:
        print(f"Error reading parquet {parquet_path}: {e}")

    # Videos
    if main_video_key and video_chunk_dir:
        src_vid = video_chunk_dir / main_video_key / f"{ep_filename}.mp4"
        if src_vid.exists():
            shutil.copy2(src_vid, dest_ep_dir / "video.mp4")
    
    if camera_map and video_chunk_dir:
        for target_name, src_key in camera_map.items():
            src_vid = video_chunk_dir / src_key / f"{ep_filename}.mp4"
            if src_vid.exists():
                shutil.copy2(src_vid, dest_raw_video_dir / f"{target_name}.mp4")
