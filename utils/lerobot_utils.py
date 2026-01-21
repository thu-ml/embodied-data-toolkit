import os
import json
import torch
import numpy as np
import shutil
from pathlib import Path

# Try importing pyarrow for parquet reading
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("Warning: 'pyarrow' library not found. Parquet reading will fail.")

def expand_single_lerobot_task(task_dir: Path, target_root: Path, params: dict):
    """
    Expands a single LeRobot task directory into Standard format.
    This function contains the core logic previously in the processor.
    """
    meta_dir = task_dir / "meta"
    
    videos_dir = task_dir / "videos"
    data_dir = task_dir / "data"
    
    # Params override
    camera_map = params.get("cameras", {})
    main_video_key = params.get("main_video_key", "observation.images.image")
    action_key = params.get("action_key", "action")
    save_action_as = params.get("save_action_as", "qpos") # qpos or eef
    
    if not meta_dir.exists():
        print(f"Meta dir not found in {task_dir}. Skipping.")
        return

    # 1. Load Metadata
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    
    if not episodes_path.exists():
        print(f"episodes.jsonl missing in {task_dir}. Skipping.")
        return
        
    # Read episodes
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
            
    # Read tasks map
    task_map = {} # task_index -> task_string
    if tasks_path.exists():
        with open(tasks_path, 'r') as f:
            for line in f:
                t = json.load(line)
                task_map[t["task_index"]] = t["task"]
    
    print(f"Processing task {task_dir.name}: {len(episodes)} episodes")

    # 2. Iterate Episodes
    for ep in episodes:
        ep_idx = ep["episode_index"]
        task_idx = ep.get("task_index", 0)
        length = ep.get("length")
        
        ep_filename = f"episode_{ep_idx:06d}"
        
        # Find chunk directory. 
        parquet_path = None
        video_chunk_dir = None 
        
        found = list(data_dir.rglob(f"{ep_filename}.parquet"))
        if found:
            parquet_path = found[0]
            chunk_name = parquet_path.parent.name
            video_chunk_dir = videos_dir / chunk_name
        else:
            # print(f"Warning: Parquet for episode {ep_idx} not found in {task_dir.name}.")
            continue
            
        # Target Directory
        dest_ep_dir = target_root / f"episode_{ep_idx}"
        dest_raw_video_dir = dest_ep_dir / "raw_video"
        dest_ep_dir.mkdir(parents=True, exist_ok=True)
        dest_raw_video_dir.mkdir(exist_ok=True)
        
        # 3. Process Instructions
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
            
        # 4. Process Actions
        if HAS_PYARROW and parquet_path:
            try:
                table = pq.read_table(parquet_path)
                df = table.to_pandas()
                
                target_col = None
                if action_key in df.columns:
                    target_col = action_key
                else:
                    for c in df.columns:
                        if 'action' in c:
                            target_col = c
                            break
                
                if target_col:
                    actions = np.stack(df[target_col].to_numpy())
                    actions = torch.from_numpy(actions).float()
                    torch.save(actions, dest_ep_dir / f"{save_action_as}.pt")
            except Exception as e:
                print(f"Error reading parquet {parquet_path}: {e}")

        # 5. Process Videos
        if main_video_key and video_chunk_dir:
            src_vid = video_chunk_dir / main_video_key / f"{ep_filename}.mp4"
            if src_vid.exists():
                shutil.copy2(src_vid, dest_ep_dir / "video.mp4")
        
        if camera_map and video_chunk_dir:
            for target_name, src_key in camera_map.items():
                src_vid = video_chunk_dir / src_key / f"{ep_filename}.mp4"
                if src_vid.exists():
                    shutil.copy2(src_vid, dest_raw_video_dir / f"{target_name}.mp4")

