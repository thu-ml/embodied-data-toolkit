from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EpisodeContext:
    # Source info
    src_episode_dir: Path
    task_name: str
    
    # Destination info
    dest_episode_dir: Path
    
    # Runtime state (mutable)
    logs: List[str] = field(default_factory=list)
    status: str = "pending" # pending, success, skipped, failed, error
    reason: Optional[str] = None
    error: Optional[str] = None
    
    # Derived paths and data
    # These might be populated by processors and shared between them
    qpos_path: Optional[Path] = None
    video_paths: Dict[str, Path] = field(default_factory=dict)
    generated_caption: Optional[str] = None
    
    def log(self, message: str):
        self.logs.append(message)
        
    def fail(self, reason: str):
        self.status = "failed"
        self.reason = reason
        self.log(f"FAILED: {reason}")
        
    def skip(self, reason: str):
        self.status = "skipped"
        self.reason = reason
        self.log(f"SKIPPED: {reason}")
        
    @property
    def episode_name(self) -> str:
        return self.src_episode_dir.name

@dataclass
class DatasetContext:
    episodes: List[EpisodeContext]
    
    def update_results(self, new_episodes: List[EpisodeContext]):
        """Update episodes with results from a processor run."""
        # Assuming order is preserved, or we map by ID. 
        # Since we pass the list and get a list, direct assignment is usually fine 
        # if the processor returns the modified context objects.
        self.episodes = new_episodes
