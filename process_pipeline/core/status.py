import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional

@dataclass
class StepStatus:
    status: str = "pending"  # pending, processing, success, failed, skipped
    timestamp: float = 0.0
    message: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EpisodeStatusData:
    steps: Dict[str, StepStatus] = field(default_factory=dict)

class EpisodeStatusManager:
    def __init__(self, episode_dir: Path, enabled: bool = True):
        self.status_file = episode_dir / ".status.json"
        self.enabled = enabled
        self._data = self._load()

    def _load(self) -> EpisodeStatusData:
        if not self.enabled:
            return EpisodeStatusData()
            
        if not self.status_file.exists():
            return EpisodeStatusData()
        
        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            steps = {}
            for step_name, step_data in raw_data.get("steps", {}).items():
                steps[step_name] = StepStatus(**step_data)
                
            return EpisodeStatusData(steps=steps)
        except Exception as e:
            print(f"Warning: Failed to load status file {self.status_file}: {e}")
            return EpisodeStatusData()

    def _save(self):
        if not self.enabled:
            return
            
        try:
            # Convert dataclass to dict
            data_dict = {
                "steps": {
                    k: asdict(v) for k, v in self._data.steps.items()
                }
            }
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save status file {self.status_file}: {e}")

    def get_step_status(self, step_name: str) -> str:
        return self._data.steps.get(step_name, StepStatus()).status

    def update_step(self, step_name: str, status: str, message: str = None, meta: dict = None):
        step = self._data.steps.get(step_name, StepStatus())
        step.status = status
        step.timestamp = time.time()
        if message is not None:
            step.message = message
        if meta:
            step.meta.update(meta)
            
        self._data.steps[step_name] = step
        self._save()
