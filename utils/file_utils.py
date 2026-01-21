import os
import re
import glob
from pathlib import Path

# --- Episode Discovery Strategies ---

def discover_episodes_directory(task_path, pattern="episode_*", id_regex=r"(\d+)"):
    """
    Strategy: Subdirectories (e.g. task/episode_0/).
    Returns: { "episode_0": { "path": Path(...), "id": "0" } }
    """
    task_path = Path(task_path)
    # Glob returns generator, convert to list
    episode_dirs = list(task_path.glob(pattern))
    
    episodes = {}
    for ep_path in episode_dirs:
        if not ep_path.is_dir():
            continue
            
        ep_dirname = ep_path.name
        
        # Extract ID
        match = re.search(id_regex, ep_dirname)
        if match:
            ep_id = match.group(1)
        else:
            # Fallback: use entire dirname if regex fails? or skip?
            # Let's use dirname for safety, but usually regex is for extracting number
            ep_id = ep_dirname
        
        std_ep_id = f"episode_{ep_id}" if not ep_id.startswith("episode_") else ep_id
        
        episodes[std_ep_id] = {
            "path": ep_path,
            "id": ep_id,
            "type": "directory"
        }
    return episodes

def discover_episodes_filename(task_path, file_pattern="*.mp4", id_regex=r"(\d+)"):
    """
    Strategy: Files (e.g. task/videos/0.mp4).
    Returns: { "episode_0": { "path": Path(task_path), "id": "0", "type": "virtual" } }
    """
    task_path = Path(task_path)
    
    # Check if pattern implies a subdirectory search (e.g. "videos/*.mp4")
    # If file_pattern is "videos/*.mp4", we glob inside task_path
    files = list(task_path.glob(file_pattern))
    
    episodes = {}
    for file_path in files:
        filename = file_path.name
        
        match = re.search(id_regex, filename)
        if not match:
            continue
            
        ep_id = match.group(1)
        std_ep_id = f"episode_{ep_id}"
        
        episodes[std_ep_id] = {
            "path": task_path, # Root is still task dir
            "id": ep_id,
            "type": "virtual",
            "anchor_file": file_path # Keep track of which file triggered this
        }
    return episodes

def discover_episodes(task_path, strategy="directory", **kwargs):
    """
    Universal entry point for discovering episodes in a task directory.
    """
    if strategy == "directory":
        pattern = kwargs.get("pattern", "episode_*")
        id_regex = kwargs.get("id_regex", r"(\d+)")
        return discover_episodes_directory(task_path, pattern, id_regex)
        
    elif strategy == "filename_match":
        primary = kwargs.get("primary_source", {})
        # Handle both config dict style and direct kwargs style
        file_pattern = primary.get("path") if isinstance(primary, dict) else kwargs.get("file_pattern", "*.mp4")
        id_regex = primary.get("id_regex") if isinstance(primary, dict) else kwargs.get("id_regex", r"(\d+)")
        
        return discover_episodes_filename(task_path, file_pattern, id_regex)
    
    return {}

def find_task_directories(root_path, task_pattern="*", episode_config=None):
    """
    Recursively find directories that contain episodes based on the config.
    
    Args:
        root_path: Source root
        task_pattern: Glob pattern for task dirs (default "*")
        episode_config: Dict with 'strategy', 'pattern', etc.
                       If None, defaults to directory strategy looking for 'episode_*'
    
    Returns:
        list of dict: [ { "path": Path, "name": str, "rel_path": Path, "episodes": dict }, ... ]
    """
    task_dirs = []
    root_path = Path(root_path)
    
    if episode_config is None:
        episode_config = {"strategy": "directory", "pattern": "episode_*"}

    # Recursive search for POTENTIAL task directories
    # Optimization: If task_pattern is simple "*", just iterate subdirs?
    # But usually we want to support nested structures like group/task.
    
    # We use glob(recursive=True)
    potential_dirs = root_path.glob(task_pattern) if "**" in task_pattern else root_path.glob(task_pattern)
    # If pattern is just "*" and we want recursive, we might need rglob or "**/*"
    # To keep it safe and robust:
    if task_pattern == "*":
        # Recursive walk manually to be safe? Or assume tasks are 1 level deep?
        # The previous implementation did a recursive walk. Let's replicate that behavior 
        # but using the new validator.
        potential_dirs = [x for x in root_path.rglob("*") if x.is_dir()]
    else:
        potential_dirs = [x for x in root_path.glob(task_pattern) if x.is_dir()]
        
    for p in potential_dirs:
        # Check if valid
        episodes = discover_episodes(p, **episode_config)
        if len(episodes) > 0:
            rel_path = p.relative_to(root_path)
            task_dirs.append({
                "path": p,
                "name": p.name,
                "rel_path": rel_path,
                "episodes": episodes
            })
            
    return task_dirs

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
