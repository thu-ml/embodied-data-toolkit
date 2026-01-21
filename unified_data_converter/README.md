
# 🤖 Unified Embodied Data Converter

**Unified Embodied Data Converter** is a core component of the Embodied Data Toolkit. It acts as a universal transformation engine, capable of converting heterogeneous raw robotics data (HDF5, Directory-based, Mixed) into any desired directory structure and format using purely declarative configuration.

---

## ✨ Key Features

- 🧩 **No-Code Mapping**: Define source data hierarchy and target structures via JSON configuration files. Adapt to new datasets without modifying core engine code.
- ⚡ **High-Performance Parallelism**: Built-in multi-process task scheduler based on `ProcessPoolExecutor`, supporting parallel conversion of large-scale datasets.
- 🛡️ **Resume Capability**: Task-hash-based caching mechanism automatically identifies processed files, allowing fast recovery after interruptions and avoiding redundant computation.
- 🛠️ **Extensible Processors**:
  - **HDF5 Expert**: Support for extracting compressed byte-stream videos from HDF5, reading specific fields, and merging Tensors.
  - **Multimedia Processing**: Integrated FFMPEG/OpenCV interfaces for multi-view video concatenation, frame extraction, and format conversion.
  - **Language Modality**: Automatic transformation of instruction JSONs, supporting sub-instruction generation and text extraction.
- 🔍 **Dynamic Context Awareness**: Support for dynamic variables like `{task_name}` and `{raw_id}` in path definitions for complex renaming logic.

---

## 📖 Configuration Guide

The configuration consists of two main sections: **Hierarchy** (Reading Source) and **Target Structure** (Writing Destination).

### 1. Hierarchy (Defining Source)
Describes how to recursively scan the raw directory.
- `directory`: Matches sub-directory names.
- `filename_match`: Uses regex to extract metadata (e.g., Episode ID) from filenames.

```json
"hierarchy": [
    { "name": "task", "type": "directory", "pattern": "*", "context_key": "task_name" },
    { "name": "episode", "type": "filename_match", "primary_source": { "path": "video/*.mp4", "id_regex": "(\\d+)" }, "context_key": "ep_id" }
]
```

### 2. Target Structure (Defining Output)
Describes the output tree.
- `iterator`: Binds a node to a source hierarchy level.
- `processor`: Specifies the python function to run for file generation.
- `params`: Supports special protocols like `src://` (Source Root) and `dest://` (Target Root).

```json
"target_structure": [
    {
        "name": "{task_name}/data/{ep_id}.mp4",
        "type": "file",
        "iterator": "episode",
        "processor": "copy",
        "params": { "source": "src://{task_name}/video/{ep_id}.mp4" }
    }
]
```

---

## 🔧 Advanced Features

### Aggregation Queries
Gather data from multiple child nodes to create a summary.
```json
"params": {
    "all_videos": {
        "from": "source",
        "select_iterator": "episode",
        "target_file": "video/*.mp4"
    }
}
```

### Custom Processors
Add your own logic in `processors/` and register it:
```python
@ProcessorRegistry.register("my_proc")
def my_proc(target_path, params, context):
    pass
```

---

## 📂 Documentation

- [**Advanced Configuration Guide**](./configs/CONFIG_GUIDE_ADVANCED.md): In-depth guide with real-world scenarios, detailed field specifications, and debugging tips.
- **Design Concepts**: See the main README for architecture overview.
