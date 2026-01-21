
# 🌊 Process Pipeline Guide

The **Process Pipeline** is a modular, high-throughput framework for processing robotic trajectories. It supports multi-level concurrency (Episode/Task/Dataset), resume-from-breakpoint (via Redis), and easy extensibility.

---

## 🚀 Quick Start

### 1. Configure the Pipeline
Edit `process_pipeline/configs/config.yaml` to define your input path and processing steps.

```yaml
# Input dataset root (Standard Data Format)
source_root: "/path/to/standard_data"

# Redis Config (Optional, for global state tracking)
redis:
  host: "localhost"
  port: 6379
  db: 0
  dataset_id: "my_dataset_v1"

# Concurrency
workers: 16

# Processing Steps (Executed sequentially)
steps:
  - name: "validation"
    processor: "validation"
    params:
      perform: true

  - name: "trimming"
    processor: "trim"
    params:
      threshold: 10
      video_trim_mode: "ffmpeg"

  - name: "captioning"
    processor: "caption"
    params:
      api_key: "sk-..."
      model: "gpt-4o"
```

### 2. Run the Pipeline
```bash
# Using Python directly
python process_pipeline/process_pipeline.py --config process_pipeline/configs/config.yaml

# Or using the wrapper script
bash data_process_1/scripts/run_process_pipeline.sh
```

---

## 🧩 Adding Custom Processors

The pipeline uses a **Plugin Architecture**. You can easily add new processors by inheriting from the appropriate base class and registering them.

### 1. Choose the Granularity
Decide at which level your processor should operate:

| Level | Base Class | Use Case |
| :--- | :--- | :--- |
| **Episode** | `EpisodeRunner` | Process each episode independently (e.g., Trimming, Image Encoding). |
| **Task** | `TaskRunner` | Aggregate episodes within a task (e.g., Train/Val split, Task-level summary). |
| **Dataset** | `DatasetRunner` | Global operation on the entire dataset (e.g., Global Statistics, Normalization). |

### 2. Implementation Example

Create a new file `process_pipeline/processors/my_processor.py`:

#### A. Episode-Level Processor
```python
from core.runners import EpisodeRunner
from core.registry import ProcessorRegistry
from core.context import EpisodeContext

@ProcessorRegistry.register("my_episode_proc")
class MyEpisodeProcessor(EpisodeRunner):
    def __init__(self, config=None):
        super().__init__(config)
        self.param1 = self.config.get("param1", "default")

    def process_episode(self, context: EpisodeContext) -> EpisodeContext:
        # Access data
        video_path = context.dest_episode_dir / "video.mp4"
        
        # Your Logic Here
        print(f"Processing {context.episode_name} with {self.param1}")
        
        # Log and return
        context.log("Processed successfully")
        return context
```

#### B. Task-Level Processor
```python
from core.runners import TaskRunner
from core.registry import ProcessorRegistry
from core.context import EpisodeContext
from typing import List

@ProcessorRegistry.register("my_task_proc")
class MyTaskProcessor(TaskRunner):
    def process_task(self, task_name: str, episodes: List[EpisodeContext]) -> List[EpisodeContext]:
        # Example: Filter episodes in a task
        print(f"Task {task_name} has {len(episodes)} episodes")
        
        # Return modified list or update properties
        return episodes
```

#### C. Hybrid Processor (Param-based Switching)
You can support multiple levels in one class by checking config parameters.

```python
from core.runners import EpisodeRunner, TaskRunner
from core.registry import ProcessorRegistry

# Register generic class
@ProcessorRegistry.register("flexible_proc")
class FlexibleProcessor:
    def __new__(cls, config=None):
        mode = config.get("mode", "episode")
        
        if mode == "task":
            return _TaskImpl(config)
        else:
            return _EpisodeImpl(config)

class _EpisodeImpl(EpisodeRunner):
    def process_episode(self, context):
        # ... logic
        return context

class _TaskImpl(TaskRunner):
    def process_task(self, task_name, episodes):
        # ... logic
        return episodes
```

### 3. Register and Use
1. Import your new processor in `process_pipeline/processors/__init__.py` so it gets registered.
   ```python
   from .my_processor import MyEpisodeProcessor
   ```
2. Add it to your `config.yaml`:
   ```yaml
   steps:
     - name: "my_custom_step"
       processor: "my_episode_proc"
       params:
         param1: "value"
   ```

---

## 🛠️ Context Object
The `context` object passed to methods contains all necessary paths and metadata.

- `context.src_episode_dir`: Path to source episode (if copying).
- `context.dest_episode_dir`: Path to current working directory for the episode.
- `context.task_name`: Name of the parent task.
- `context.episode_name`: ID/Name of the episode.
- `context.metadata`: Dictionary for storing intermediate results across steps.


