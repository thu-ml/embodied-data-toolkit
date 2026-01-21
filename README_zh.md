# Embodied Data Toolkit (具身智能数据工具箱)

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**Embodied Data Toolkit** 是一个为具身智能和机器人学习设计的数据处理和格式转换框架。它提供了从原始数据摄取、格式转换到高级轨迹处理的完整解决方案。

该工具箱由两个核心组件组成：
1.  **Unified Data Converter (统一数据转换器)**：一个配置驱动的引擎，用于将异构原始数据（HDF5、Pytorch Tensor、Json、mp4 等）转换为任何指定的格式。
2.  **Process Pipeline (处理流水线)**：一个模块化的轨迹处理工作流管理器（支持裁剪、标注、拼接等），内置断点续传功能。

---

## 🏗️ 架构

该框架采用分层处理架构，以确保高吞吐量和可靠性。

![架构图](./assets/process_pipeline.png)

### 1. Unified Data Converter (格式引擎)
- **无代码映射**：通过 JSON 配置定义源到目标的映射，无需编写代码。
- **协议支持**：原生支持 `src://`（源根目录）和 `dest://`（目标根目录）协议。
- **多媒体专家**：从 HDF5 中提取压缩视频、合并张量，并处理多模态数据。
- **高级聚合**：能够跨逻辑层级查询和聚合数据（例如：收集一个任务下的所有回合以生成任务摘要）。

### 2. Process Pipeline (工作流管理器)
- **多级并发**：使用 `multiprocessing` 在 Episode（回合）、Task（任务）和 Dataset（数据集）级别进行并行处理。
- **可恢复执行**：使用 Redis 和本地 `.status.json` 文件跟踪进度，实现断点续传。
- **可插拔步骤**：内置 **Validation** (校验)、**Structure** (结构化)、**Concat** (拼接)、**Caption** (标注) 和 **Trim** (裁剪) 等处理器。

---

## 🚀 快速开始

### 1. 安装
```bash
git clone https://github.com/thu-ml/embodied-data-toolkit.git
cd embodied-data-toolkit

conda create -n embodied-data-toolkit python==3.10
conda activate embodied-data-toolkit

pip install -r requirements.txt
# 确保系统已安装 ffmpeg
# sudo apt install ffmpeg
```

### 2. 组件使用

#### A. 数据转换
使用 JSON 配置将原始数据集转换为标准结构（请先定义对应的配置 JSON）：
```bash
python unified_data_converter/run_conversion.py \
    --config unified_data_converter/configs/my_config.json \
    --src_root /path/to/raw_data \
    --dest_root /path/to/standard_data \
    --workers 16
```

或者使用脚本：
```bash
bash scripts/run_conversion.sh
```

#### B. 轨迹处理
运行高级处理流水线（裁剪、标注等），修改 `config.yaml` 并添加更多处理器以适应您的流程：
```bash
python process_pipeline/process_pipeline.py \
  --config process_pipeline/configs/config.yaml
```

或者使用脚本：
```bash
bash scripts/run_process_pipeline.sh
```

---

## 📂 数据格式

### 1. Process Pipeline 输入格式

```text
dataset_root/
├── folder_1/
│   └── ...
└── folder_n/
    └── {task_name}/
        ├── episode_0/
        │   ├── episode_0_cam_front.mp4      # 前视角视频（固定后置摄像头，弃用）
        │   ├── episode_0_cam_high.mp4       # 高位摄像头视频
        │   ├── episode_0_cam_left_wrist.mp4  # 左臂夹爪摄像头视频
        │   ├── episode_0_cam_right_wrist.mp4 # 右臂夹爪摄像头视频
        │   ├── episode_0_qpos.pt            # 关节位置序列 (T, 14)
        │   └── episode_0_tts.mp4            # 音频/TTS (可选)
        ├── episode_1/
        │   └── ...
        └── ...
```

### 2. 标准数据格式 (Pipeline 输出)
**Process Pipeline** 处理上述输入（裁剪、拼接、标注等），并生成最终用于训练的标准结构。

```text
{task_name}/
├── task_meta.json           # 全局元数据和任务级指令
└── episode_{id}/            # 单个回合目录
    ├── video.mp4            # 主视频/合并视频 (Concat 处理器的结果，顶部为 cam_high.mp4，左下为 cam_left_wrist.mp4，右下为 cam_right_wrist.mp4)
    ├── qpos.pt              # 关节位置和夹爪状态 (torch.Tensor)
    ├── endpose.pt           # 末端执行器笛卡尔位姿 (可选, torch.Tensor)
    ├── instructions.json    # 语言元数据 (总帧数, 指令, 分段)
    ├── umt5_wan/            # (可选示例) 语言嵌入 (UMT5/Wan2.2)
    └── raw_video/           # 原始摄像头视角
        ├── cam_high.mp4     # 固定高视角 (例如：顶部/后方)
        ├── cam_left_wrist.mp4
        ├── cam_right_wrist.mp4
        └── cam_front.mp4    # (可选) 前方/侧方视角
```

### 关键数据规范
- **张量 (Tensors)**：`.pt` 文件应通过 `torch.save()` 保存。
- **视频 (Videos)**：`.mp4` 文件建议使用 H.264 编码以获得最佳兼容性。
- **指令 (Instructions)**：`instructions.json` 应至少包含顶层的 `instructions` 字符串列表和帧级子指令。

```json
{
  "instructions": ["aaa","bbb","ccc"],
  "sub_instructions": [
    {"start_frame": 0, "end_frame": 150, "instruction": ["aaa"]},
    {"start_frame": 150, "end_frame": 340, "instruction": ["bbb", "ccc"]}
  ]
}
```

---

## 🛠️ Redis 管理

**Process Pipeline** 使用 Redis 维护全局状态，以支持断点续传（检查点）。

### 安装 (Linux/Ubuntu)
```bash
sudo apt update
sudo apt install redis-server
```

### 启动 Redis
- **作为系统服务 (推荐)**：
  ```bash
  sudo systemctl start redis-server
  # 设置开机自启
  sudo systemctl enable redis-server
  ```
- **在后台手动启动**：
  ```bash
  redis-server --daemonize yes
  ```

### 停止 Redis
- **作为系统服务**：
  ```bash
  sudo systemctl stop redis-server
  ```
- **手动停止**：
  ```bash
  redis-cli shutdown
  ```

### 检查状态
```bash
redis-cli ping
# 应返回 "PONG"
```

---

## 🧩 处理器详情 (部分)

| 组件 | 处理器 | 描述 | 关键参数 |
| :--- | :--- | :--- | :--- |
| **Pipeline** | **Validation** | 验证数据完整性和合规性 | `perform: true` |
| **Pipeline** | **Structure** | 重新构建目录层级 | `fast_video_copy` |
| **Pipeline** | **Concat** | 合并多视角视频 (Top/Left/Right) | `fps` |
| **Pipeline** | **Caption** | 生成文本描述 (GPT/VLM) | `api_key`, `system_prompt` |
| **Pipeline** | **Trim** | 基于动作裁剪静态帧 | `threshold`, `video_trim_mode` |
| **Converter** | **copy** | 简单的文件复制 | `source` |
| **Converter** | **hdf5_extractor** | 从 HDF5 文件提取数据 | `source_h5`, `fields` |
| **Converter** | **json_transformer** | 转换 JSON 结构 | `template` |

---

## 📂 项目结构

```text
.
├── unified_data_converter/   # 格式转换引擎
│   ├── configs/              # JSON 转换规则
│   ├── core/                 # 解析器、规划器、上下文
│   ├── processors/           # HDF5、视频、JSON 转换器
│   └── run_conversion.py     # 入口点
├── process_pipeline/         # 工作流与轨迹管理器
│   ├── configs/              # 流水线 YAML 配置
│   ├── core/                 # 流水线与运行器 (Episode/Task)
│   ├── processors/           # 裁剪、标注、拼接步骤
│   └── process_pipeline.py   # 入口点
├── utils/                    # 共享的 IO、视频和张量工具
└── README.md
```

---

## ⚠️ 注意事项

1.  **并发性**：两个组件都支持 `--workers` 或配置中的 `workers` 来调整 CPU 使用率。
2.  **裁剪模式 (Trim Mode)**：
    - `ffmpeg`：高质量，速度较慢。
    - `fast` (OpenCV)：高速度 (快 8-10 倍)，文件体积较大。
3.  **HDF5 依赖**：如果使用 `hdf5_extractor`，请确保已安装 `h5py`。

