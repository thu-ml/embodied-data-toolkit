# 🛠️ 统一数据转换器配置指南

本指南详细解析了配置字段，并提供了一系列真实场景的解决方案，帮助您精通用于复杂机器人数据转换的 `config.json`。

---

## 1. 核心逻辑

配置文件由两个主要部分组成：

1.  **`hierarchy` (源解析)**：定义“源数据长什么样”以及如何遍历目录结构。
2.  **`target_structure` (目标生成)**：定义“你想生成什么”，指定输出目录树和处理逻辑。

---

## 2. Hierarchy：定义源结构

`hierarchy` 是一个按深度顺序定义的列表。每个层级都会生成一个 **Iterator (迭代器)** 供后续使用。

### 2.1 常见模式

#### 模式 A：目录遍历 (`directory`)
最适合标准的嵌套文件夹结构。

```json
{
    "name": "task",           // 迭代器名称
    "type": "directory",      // 类型：目录
    "pattern": "*",           // Glob 模式：匹配所有文件夹
    "context_key": "task_name" // 将目录名存储在变量 {task_name} 中
}
```

#### 模式 B：文件名匹配与 ID 提取 (`filename_match`)
最适合扁平结构或需要从文件名中提取 ID 的情况。

```json
{
    "name": "episode",
    "type": "filename_match",
    "primary_source": {
        "path": "video/episode*.mp4", // 搜索匹配此模式的文件
        "id_regex": "episode(\\d+)"   // 正则提取：() 中的内容成为 ID
    },
    "context_key": "ep_id"            // 将提取的 ID (例如 "0") 存储在 {ep_id} 中
}
```

### 2.2 示例：源结构

**结构 1：三层嵌套 (Category -> Task -> Episode)**
```text
Root/
  ├── manipulation/ (Category)
  │    ├── pick_apple/ (Task)
  │    │    ├── episode_0/ (Episode)
```
**配置：**
```json
"hierarchy": [
    { 
        "name": "category", 
        "type": "directory", 
        "pattern": "*", 
        "context_key": "cat" 
    },
    { 
        "name": "task", 
        "type": "directory", 
        "pattern": "*", 
        "context_key": "task" 
    },
    { 
        "name": "episode", 
        "type": "directory", 
        "pattern": "episode_*", 
        "context_key": "ep_dir" 
    }
]
```

**结构 2：扁平任务 + 文件 (Task -> File)**
```text
Root/
  ├── pick_apple/ (Task)
  │    ├── episode_0.hdf5
  │    ├── episode_1.hdf5
```
**配置：**
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
            "path": "*.hdf5", 
            "id_regex": "episode_(\\d+)" 
        }, 
        "context_key": "ep_id" 
    }
]
```

---

## 3. Target Structure：定义输出与逻辑

这是一个递归树结构，定义了生成规则。

### 3.1 关键字段

*   **`name`**：
    *   目标文件名或目录名。
    *   支持变量替换：`"episode_{ep_id}.mp4"`。
    *   **特殊用法**：`"name": "."` —— **透明层**。不会创建物理目录，仅用于迭代逻辑和上下文更新。
*   **`type`**：`"directory"` 或 `"file"`。
*   **`iterator`**：
    *   绑定到 `hierarchy` 中定义的层级。
    *   对于在该层级找到的每个项目，都会重复生成该节点。
*   **`processor`**：（仅限文件）指定要执行的 Python 函数（例如 `copy`, `extract_from_hdf5`）。
*   **`params`**：传递给处理器的参数字典。

### 3.2 路径协议

在 `params` 中引用路径时使用以下协议：
*   **`src://`**：从 **源根目录 (Source Root)** 开始的绝对路径。
    *   示例：`src://{task_name}/video/{ep_id}.mp4`
*   **`dest://`**：从 **目标根目录 (Target Root)** 开始的绝对路径（引用已生成的文件）。
    *   示例：`dest://{task_name}/processed_video.mp4`

---

## 4. 示例：真实场景方案

### 场景 1：扁平化结构
**目标**：将 `Category/Task/Episode/video.mp4` 转换为扁平文件夹中的 `Category_Task_Episode.mp4`。

**源结构**：
```text
Category/
  Task/
    ep0/video.mp4
```

**配置**：
```json
"target_structure": [
    {
        "name": ".",              // 透明层：迭代 Category
        "iterator": "category",
        "type": "directory",
        "children": [
            {
                "name": ".",          // 透明层：迭代 Task
                "iterator": "task",
                "type": "directory",
                "children": [
                    {
                        "name": "{cat}_{task}_{ep_id}.mp4", // 扁平化命名
                        "type": "file",
                        "iterator": "episode",
                        "processor": "copy",
                        "params": {
                            "source": "src://{cat}/{task}/{ep_dir}/video.mp4"
                        }
                    }
                ]
            }
        ]
    }
]
```

### 场景 2：HDF5 数据提取
**目标**：从单个 HDF5 文件中提取视频流和动作数组。

**源文件**：`data.hdf5` (包含键：`obs/images/cam_high`, `action`)

**配置**：
```json
{
    "name": "cam_high.mp4",
    "type": "file",
    "processor": "extract_from_hdf5",
    "params": {
        "source": "src://{task}/data.hdf5",
        "key": "obs/images/cam_high",
        "format": "video",
        "fps": 30,
        "is_video_bytes": true // 如果 HDF5 存储的是 JPEG 字节
    }
},
{
    "name": "action.npy",
    "type": "file",
    "processor": "extract_from_hdf5",
    "params": {
        "source": "src://{task}/data.hdf5",
        "key": "action",
        "format": "npy"
    }
}
```

### 场景 3：多视角视频拼接
**目标**：先提取 3 个视角，然后将它们拼接成一个宽视角视频。使用 `dest://` 引用刚刚生成的文件。

**配置**：
```json
{
    "name": "raw_videos", // 为原始视角创建文件夹
    "type": "directory",
    "children": [
        { 
            "name": "left.mp4", 
            "processor": "copy", 
            ... 
        },
        { 
            "name": "right.mp4", 
            "processor": "copy",
            ... 
        },
        { 
            "name": "top.mp4", 
            "processor": "copy", 
            ... 
        }
    ]
},
{
    "name": "merged_video.mp4", // raw_videos 的兄弟节点
    "type": "file",
    "processor": "concat_video_3views",
    "params": {
        "cam_left_wrist": "dest://{task}/raw_videos/left.mp4",
        "cam_right_wrist": "dest://{task}/raw_videos/right.mp4",
        "cam_high": "dest://{task}/raw_videos/top.mp4"
    }
}
```

### 场景 4：LeRobot 格式转换 (聚合)
**目标**：不是逐个文件转换，而是触发库调用将整个 Task 转换为 LeRobot 数据集。

**配置**：
```json
{
    "name": "{task_name}_conversion.log", // 标记文件
    "type": "file",
    "iterator": "task", // 在 Task 层级迭代
    "processor": "convert_task_to_lerobot", // 聚合处理器
    "params": {
        "source_task_dir": "src://{task_name}", // 传递完整的源路径
        "repo_id": "lerobot/{task_name}",
        "cameras": {
            "image": "raw_video/cam_high.mp4",
            "wrist": "raw_video/cam_wrist.mp4"
        }
    }
}
```

### 场景 5：指令提取与清洗
**目标**：源元数据是复杂的 JSON；仅提取第一个指令字符串到 TXT 文件中。

**配置**：
```json
{
    "name": "language_instruction.txt",
    "type": "file",
    "processor": "extract_first_instruction_to_txt",
    "params": {
        "source": "src://{task}/{ep_id}/metadata.json"
    }
}
```

---

## 5. 调试技巧

如果您的转换导致 **0 jobs generated**，通常意味着 `hierarchy` 未能匹配源目录结构。

1.  **检查模式 (Patterns)**：验证 glob 模式（例如 `episode_*` vs `episode*`）。
2.  **检查深度 (Depth)**：确保配置与实际深度匹配（例如 3 层 vs 2 层）。
3.  **优先使用目录 (Directories)**：尽可能使用带有 `context_key` 的 `type: directory`。对于结构化数据集，它比 `filename_match` 更稳健。

