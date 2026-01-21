import os
import cv2
import torch
import json
import shutil
import numpy as np
from pathlib import Path
from core.registry import ProcessorRegistry

# Import shared utils
try:
    from utils.video_io import get_video_info, read_video_frames
except ImportError:
    # Fallback if utils not in path or running standalone
    print("Warning: Could not import utils.video_io in lerobot.py")
    pass

# Try importing lerobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("Warning: 'lerobot' library not found. LeRobot processors will not work.")

@ProcessorRegistry.register("convert_task_to_lerobot")
def process_convert_task_to_lerobot(target_path: str, params: dict, context: dict):
    """
    Converts a standard task directory (Task/Episode/...) to LeRobot dataset format.
    
    Target Path: Ideally the output directory for the LeRobot dataset.
    
    Params:
        source_task_dir (str): Path to the source task directory containing episode subdirs.
        repo_id (str): LeRobot repo ID (e.g. "user/dataset_name").
        robot_type (str): Robot type string (default "aloha").
        cameras (dict): Mapping of LeRobot camera keys to source filenames.
                        e.g. {"image": "cam_high.mp4", "wrist_image": "cam_left_wrist.mp4"}
    """
    if not HAS_LEROBOT:
        raise ImportError("LeRobot library is required for this processor.")

    source_task_dir = params.get("source_task_dir")
    repo_id = params.get("repo_id")
    robot_type = params.get("robot_type", "aloha")
    camera_map = params.get("cameras", {
        "image": "cam_high.mp4",
        "left_wrist_image": "cam_left_wrist.mp4",
        "right_wrist_image": "cam_right_wrist.mp4"
    })
    
    if not source_task_dir or not os.path.exists(source_task_dir):
        raise FileNotFoundError(f"Source task dir not found: {source_task_dir}")
        
    task_name = os.path.basename(source_task_dir)
    print(f"[LeRobot] Processing Task: {task_name} -> {repo_id}")
    
    # 1. Scan Episodes
    # Standard format: episode_0, episode_1...
    src_path = Path(source_task_dir)
    episode_dirs = []
    for p in src_path.iterdir():
        if p.is_dir() and "episode_" in p.name:
            try:
                # Try to parse index for sorting
                idx_str = p.name.split('_')[-1]
                idx = int(idx_str)
                episode_dirs.append((idx, p))
            except ValueError:
                continue
    
    episode_dirs.sort(key=lambda x: x[0])
    episode_dirs = [x[1] for x in episode_dirs]
    
    if not episode_dirs:
        print(f"[LeRobot] No episodes found in {source_task_dir}. Skipping.")
        return

    # 2. Determine Features from first episode
    first_ep = episode_dirs[0]
    
    # Check cameras
    active_cameras = {}
    main_fps = 30
    
    # raw_video path assumption: episode_dir/raw_video/file.mp4
    # OR standard format might be episode_dir/videos/file.mp4?
    # Let's check common locations
    
    def find_video(ep_path, filename):
        candidates = [
            ep_path / "raw_video" / filename,
            ep_path / "videos" / filename,
            ep_path / filename
        ]
        for c in candidates:
            if c.exists(): return c
        return None

    for key, filename in camera_map.items():
        v_path = find_video(first_ep, filename)
        if v_path:
            fps, w, h = get_video_info(v_path)
            active_cameras[key] = {
                "filename": filename, 
                "width": w, 
                "height": h,
                "fps": fps
            }
            if key == "image": main_fps = fps

    if not active_cameras:
        print(f"[LeRobot] No valid videos found for {task_name}. Skipping.")
        return

    # Check Action Dim (qpos.pt)
    qpos_path = first_ep / "qpos.pt"
    if not qpos_path.exists():
        # Try qpos/0.pt structure if standard format changed
        qpos_path = first_ep / "qpos" / f"{first_ep.name.split('_')[-1]}.pt"
        
    if not qpos_path.exists():
         print(f"[LeRobot] No qpos.pt found for {task_name}. Skipping.")
         return
         
    qpos_data = torch.load(qpos_path)
    # Handle shape (T, Dim) or (Dim,)
    if qpos_data.ndim == 1:
        action_dim = qpos_data.shape[0]
    else:
        action_dim = qpos_data.shape[1]

    # 3. Init Dataset
    features = {
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    for cam_key, info in active_cameras.items():
        features[cam_key] = {
            "dtype": "video",
            "shape": (info["height"], info["width"], 3),
            "names": ["height", "width", "channel"],
        }
    
    
    dataset_root = Path(target_path).parent
    full_repo_path = dataset_root / repo_id
    
    if full_repo_path.exists():
        shutil.rmtree(full_repo_path)

    
    # Let's double check.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=int(main_fps),
        root=full_repo_path,  # <--- Pass the FULL path so LeRobot uses it directly
        use_videos=True,
        features=features,
        image_writer_threads=8, 
        image_writer_processes=0, 
    )

    # 4. Process Episodes
    for ep_dir in episode_dirs:
        try:
            # Load QPOS
            # Try flat file first
            qp_p = ep_dir / "qpos.pt"
            if not qp_p.exists():
                 # Try nested
                 ep_idx = ep_dir.name.split('_')[-1]
                 qp_p = ep_dir / "qpos" / f"{ep_idx}.pt"
            
            qpos = torch.load(qp_p).float().numpy()
            
            # Load Instructions
            task_desc = f"Perform {task_name}"
            # Try reading instructions.json
            instr_p = ep_dir / "instructions.json"
            if instr_p.exists():
                with open(instr_p, 'r') as f:
                    d = json.load(f)
                    if "instructions" in d and d["instructions"]:
                        task_desc = d["instructions"][0]
                    elif "seen" in d and d["seen"]:
                         task_desc = d["seen"][0]
            
            # Load Videos
            video_frames = {}
            lengths = [len(qpos)]
            
            for cam_key, info in active_cameras.items():
                v_path = find_video(ep_dir, info["filename"])
                if v_path:
                    frames = read_video_frames(v_path)
                    video_frames[cam_key] = frames
                    lengths.append(len(frames))
                else:
                    raise FileNotFoundError(f"Video {info['filename']} missing in {ep_dir}")
            
            # Align
            min_len = min(lengths)
            qpos = qpos[:min_len]
            
            # Add Frames
            for i in range(min_len):
                frame = {"actions": qpos[i]}
                for k, v in video_frames.items():
                    frame[k] = v[i]
                dataset.add_frame(frame, task=task_desc)
                
            dataset.save_episode()
            
        except Exception as e:
            print(f"Error in episode {ep_dir.name}: {e}")
            continue

    dataset.consolidate()
    
    # Write a marker file to target_path to satisfy the converter engine
    with open(target_path, 'w') as f:
        f.write(f"Converted to LeRobot Repo: {repo_id}")
