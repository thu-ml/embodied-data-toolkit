# 🤖 统一具身智能数据转换器 (Unified Embodied Data Converter)

**Unified Embodied Data Converter** 是 Embodied Data Toolkit 的核心组件。它作为一个通用的转换引擎，能够通过纯声明式配置，将异构的原始机器人数据（HDF5、基于目录的数据、混合格式）转换为任何所需的目录结构和格式。

---

## ✨ 关键特性

- 🧩 **无代码映射**：通过 JSON 配置文件定义源数据层级和目标结构。无需修改核心引擎代码即可适配新数据集。
- ⚡ **高性能并行**：内置基于 `ProcessPoolExecutor` 的多进程任务调度器，支持大规模数据集的并行转换。
- 🛡️ **续传能力**：基于任务哈希（Task-hash）的缓存机制自动识别已处理的文件，允许在中断后快速恢复，避免重复计算。
- 🛠️ **可扩展处理器**：
  - **HDF5 专家**：支持从 HDF5 中提取压缩字节流视频、读取特定字段以及合并张量。
  - **多媒体处理**：集成 FFMPEG/OpenCV 接口，用于多视角视频拼接、帧提取和格式转换。
  - **语言模态**：自动转换指令 JSON，支持子指令生成和文本提取。
- 🔍 **动态上下文感知**：支持在路径定义中使用 `{task_name}` 和 `{raw_id}` 等动态变量，实现复杂的重命名逻辑。

---

## 📖 配置指南

配置由两个主要部分组成：**Hierarchy (层级)**（读取源）和 **Target Structure (目标结构)**（写入目的地）。

### 1. Hierarchy (定义源)
描述如何递归扫描原始目录。
- `directory`：匹配子目录名称。
- `filename_match`：使用正则表达式从文件名中提取元数据（如 Episode ID）。

```json
"hierarchy": [
    { 
        "name": "task", 
        "type": "directory", 
        "pattern": "*", 
        "context_key": "task_name" 
    },
    { 
        "name": "episode", 
        "type": "filename_match", 
        "primary_source": { 
            "path": "video/*.mp4", 
            "id_regex": "(\\d+)" 
        }, 
        "context_key": "ep_id" 
    }
]
```

### 2. Target Structure (定义输出)
描述输出树结构。
- `iterator`：将节点绑定到源层级（Hierarchy）的某个级别。
- `processor`：指定用于生成文件的 Python 函数。
- `params`：支持特殊协议，如 `src://`（源根目录）和 `dest://`（目标根目录）。

```json
"target_structure": [
    {
        "name": "{task_name}/data/{ep_id}.mp4",
        "type": "file",
        "iterator": "episode",
        "processor": "copy",
        "params": { 
            "source": "src://{task_name}/video/{ep_id}.mp4"
        }
    }
]
```

---

## 🔧 高级功能

### 聚合查询 (Aggregation Queries)
从多个子节点收集数据以创建摘要。
```json
"params": {
    "all_videos": {
        "from": "source",
        "select_iterator": "episode",
        "target_file": "video/*.mp4"
    }
}
```

### 自定义处理器 (Custom Processors)
在 `processors/` 中添加您自己的逻辑并注册它：
```python
@ProcessorRegistry.register("my_proc")
def my_proc(target_path, params, context):
    pass
```

---

## 📂 文档

- [**配置指南**](./configs/CONFIG_GUIDE_ADVANCED.md)：深入解析字段规格、实际场景和调试技巧。
- **设计理念**：请参阅主目录下的 README 以了解架构概览。

