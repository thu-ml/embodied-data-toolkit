# 🌊 处理流水线指南 (Process Pipeline Guide)

**Process Pipeline** 是一个模块化、高吞吐量的机器人轨迹处理框架。它支持多级并发（Episode/Task/Dataset）、基于 Redis 的断点续传，并且易于扩展。

---

## 🚀 快速开始

### 1. 配置流水线
编辑 `process_pipeline/configs/config.yaml` 来定义您的输入路径和处理步骤。

```yaml
# 输入数据集根目录 (符合标准数据格式)
source_root: "/path/to/standard_data"

# Redis 配置 (可选，用于全局状态跟踪)
redis:
  host: "localhost"
  port: 6379
  db: 0
  dataset_id: "my_dataset_v1"

# 并发设置
workers: 16

# 处理步骤 (按顺序执行)
steps:
  - name: "validation"
    processor: "validation"
    params:
      perform: true

  - name: "trimming"
    processor: "trim"
    params:
      threshold: 0.01
      video_trim_mode: "ffmpeg"

  - name: "captioning"
    processor: "caption"
    params:
      api_key: "sk-..."
      model: "gpt-4o"
```

### 2. 运行流水线
```bash
# 直接使用 Python
python process_pipeline/process_pipeline.py --config process_pipeline/configs/config.yaml

# 或使用封装脚本
bash data_process_1/scripts/run_process_pipeline.sh
```

---

## 🧩 添加自定义处理器

流水线采用 **插件架构**。您可以通过继承相应的基类并注册它们来轻松添加新的处理器。

### 1. 选择粒度
决定您的处理器应在哪个层级运行：

| 层级 | 基类 | 使用场景 |
| :--- | :--- | :--- |
| **Episode (回合)** | `EpisodeRunner` | 独立处理每个回合 (例如：裁剪、图像编码)。 |
| **Task (任务)** | `TaskRunner` | 聚合任务内的所有回合 (例如：训练/验证集划分、任务级摘要)。 |
| **Dataset (数据集)** | `DatasetRunner` | 对整个数据集进行全局操作 (例如：全局统计、标准化)。 |

### 2. 实现示例

创建一个新文件 `process_pipeline/processors/my_processor.py`：

#### A. Episode 级处理器
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
        # 访问数据
        video_path = context.dest_episode_dir / "video.mp4"
        
        # 您的逻辑
        print(f"正在处理 {context.episode_name}，参数为 {self.param1}")
        
        # 记录日志并返回
        context.log("处理成功")
        return context
```

#### B. Task 级处理器
```python
from core.runners import TaskRunner
from core.registry import ProcessorRegistry
from core.context import EpisodeContext
from typing import List

@ProcessorRegistry.register("my_task_proc")
class MyTaskProcessor(TaskRunner):
    def process_task(self, task_name: str, episodes: List[EpisodeContext]) -> List[EpisodeContext]:
        # 示例：过滤任务中的回合
        print(f"任务 {task_name} 共有 {len(episodes)} 个回合")
        
        # 返回修改后的列表或更新属性
        return episodes
```

#### C. 混合处理器 (基于参数切换)
您可以在一个类中通过检查配置参数来支持多个层级。

```python
from core.runners import EpisodeRunner, TaskRunner
from core.registry import ProcessorRegistry

# 注册通用类
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
        # ... 逻辑
        return context

class _TaskImpl(TaskRunner):
    def process_task(self, task_name, episodes):
        # ... 逻辑
        return episodes
```

### 3. 注册并使用
1. 在 `process_pipeline/processors/__init__.py` 中导入您的新处理器，以便其被注册。
   ```python
   from .my_processor import MyEpisodeProcessor
   ```
2. 将其添加到您的 `config.yaml` 中：
   ```yaml
   steps:
     - name: "my_custom_step"
       processor: "my_episode_proc"
       params:
         param1: "value"
   ```

---

## 🛠️ Context 对象
传递给方法的 `context` 对象包含所有必要的路径和元数据。

- `context.src_episode_dir`：源回合目录路径（如果涉及复制）。
- `context.dest_episode_dir`：当前回合的工作目录路径。
- `context.task_name`：父任务名称。
- `context.episode_name`：回合 ID/名称。
- `context.metadata`：用于在不同步骤之间存储中间结果的字典。

