import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class RedisConfig:
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

@dataclass
class GlobalConfig:
    src: str
    dest: str
    workers: int = 8
    enable_local_status: bool = True
    mode: str = "process"  # process or clear

@dataclass
class StepConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    steps: List[StepConfig] = field(default_factory=list)
    global_cfg: GlobalConfig = field(default_factory=lambda: GlobalConfig(src="", dest=""))
    redis_cfg: RedisConfig = field(default_factory=RedisConfig)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        global_data = data.get('global', {})
        redis_data = data.get('redis', {})
        steps_data = data.get('pipeline_steps', [])
        
        steps = []
        for step in steps_data:
            steps.append(StepConfig(
                name=step['name'],
                params=step.get('params', {})
            ))
            
        return cls(
            steps=steps,
            global_cfg=GlobalConfig(**global_data),
            redis_cfg=RedisConfig(**redis_data)
        )
