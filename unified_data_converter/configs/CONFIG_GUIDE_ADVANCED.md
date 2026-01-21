# 🛠️ Unified Data Converter Configuration Guide (Advanced)

This guide provides a comprehensive breakdown of the configuration fields and a cookbook of real-world scenarios to master `config.json` for complex robotics data transformations.

---

## 1. Core Logic

The configuration file consists of two primary sections:

1.  **`hierarchy` (Source Parsing)**: Defines "what the source data looks like" and how to traverse the directory structure.
2.  **`target_structure` (Target Generation)**: Defines "what you want to generate", specifying the output directory tree and processing logic.

---

## 2. Hierarchy: Defining Source Structure

`hierarchy` is a list defined in order of depth. Each level generates an **Iterator** for subsequent use.

### 2.1 Common Modes

#### Mode A: Directory Traversal (`directory`)
Best for standard nested folder structures.

```json
{
    "name": "task",           // Iterator name
    "type": "directory",      // Type: directory
    "pattern": "*",           // Glob pattern: match all folders
    "context_key": "task_name" // Store directory name in variable {task_name}
}
```

#### Mode B: Filename Matching & ID Extraction (`filename_match`)
Best for flat structures or when you need to extract IDs from filenames.

```json
{
    "name": "episode",
    "type": "filename_match",
    "primary_source": {
        "path": "video/episode*.mp4", // Search for files matching this pattern
        "id_regex": "episode(\\d+)"   // Regex extraction: content in () becomes the ID
    },
    "context_key": "ep_id"            // Store extracted ID (e.g., "0") in {ep_id}
}
```

### 2.2 Examples: Source Structures

**Structure 1: Three-Level Nesting (Category -> Task -> Episode)**
```text
Root/
  ├── manipulation/ (Category)
  │    ├── pick_apple/ (Task)
  │    │    ├── episode_0/ (Episode)
```
**Config:**
```json
"hierarchy": [
    { "name": "category", "type": "directory", "pattern": "*", "context_key": "cat" },
    { "name": "task", "type": "directory", "pattern": "*", "context_key": "task" },
    { "name": "episode", "type": "directory", "pattern": "episode_*", "context_key": "ep_dir" }
]
```

**Structure 2: Flat Task + Files (Task -> File)**
```text
Root/
  ├── pick_apple/ (Task)
  │    ├── episode_0.hdf5
  │    ├── episode_1.hdf5
```
**Config:**
```json
"hierarchy": [
    { "name": "task", "type": "directory", "pattern": "*", "context_key": "task_name" },
    { 
        "name": "episode", 
        "type": "filename_match", 
        "primary_source": { "path": "*.hdf5", "id_regex": "episode_(\\d+)" }, 
        "context_key": "ep_id" 
    }
]
```

---

## 3. Target Structure: Defining Output & Logic

This is a recursive tree structure defining the generation rules.

### 3.1 Key Fields

*   **`name`**: 
    *   Target filename or directory name.
    *   Supports variable substitution: `"episode_{ep_id}.mp4"`.
    *   **Special Use**: `"name": "."` —— **Transparent Layer**. Does not create a physical directory, used only for iteration logic and context updates.
*   **`type`**: `"directory"` or `"file"`.
*   **`iterator`**: 
    *   Binds to a level defined in `hierarchy`.
    *   The node will be generated repeatedly for every item found in that level.
*   **`processor`**: (Files only) Specifies the Python function to execute (e.g., `copy`, `extract_from_hdf5`).
*   **`params`**: A dictionary of parameters passed to the processor.

### 3.2 Path Protocols

Use these protocols when referencing paths in `params`:
*   **`src://`**: Absolute path starting from the **Source Root**.
    *   Example: `src://{task_name}/video/{ep_id}.mp4`
*   **`dest://`**: Absolute path starting from the **Target Root** (referencing generated files).
    *   Example: `dest://{task_name}/processed_video.mp4`

---

## 4. Cookbook: Real-World Scenarios

### Scenario 1: Flattening Structure
**Goal**: Convert `Category/Task/Episode/video.mp4` to `Category_Task_Episode.mp4` in a flat folder.

**Source**:
```text
Food/
  Eat/
    ep0/video.mp4
```

**Config**:
```json
"target_structure": [
    {
        "name": ".",              // Transparent: Iterate Category
        "iterator": "category",
        "type": "directory",
        "children": [
            {
                "name": ".",          // Transparent: Iterate Task
                "iterator": "task",
                "type": "directory",
                "children": [
                    {
                        "name": "{cat}_{task}_{ep_id}.mp4", // Flat naming
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

### Scenario 2: HDF5 Data Extraction
**Goal**: Extract video stream and action arrays from a monolithic HDF5 file.

**Source**: `data.hdf5` (contains keys: `obs/images/cam_high`, `action`)

**Config**:
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
        "is_video_bytes": true // If HDF5 stores JPEG bytes
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

### Scenario 3: Multi-View Video Concatenation
**Goal**: First extract 3 views, then concatenate them into a single wide video. Using `dest://` to reference just-generated files.

**Config**:
```json
{
    "name": "raw_videos", // Create folder for raw views
    "type": "directory",
    "children": [
        { "name": "left.mp4", "processor": "copy", ... },
        { "name": "right.mp4", "processor": "copy", ... },
        { "name": "top.mp4", "processor": "copy", ... }
    ]
},
{
    "name": "merged_video.mp4", // Sibling of raw_videos
    "type": "file",
    "processor": "concat_video_3views",
    "params": {
        "cam_left_wrist": "dest://{task}/raw_videos/left.mp4",
        "cam_right_wrist": "dest://{task}/raw_videos/right.mp4",
        "cam_high": "dest://{task}/raw_videos/top.mp4"
    }
}
```

### Scenario 4: LeRobot Format Conversion (Aggregation)
**Goal**: Instead of file-by-file conversion, trigger a library call to convert an entire Task into a LeRobot Dataset.

**Config**:
```json
{
    "name": "{task_name}_conversion.log", // Marker file
    "type": "file",
    "iterator": "task", // Iterate at Task level
    "processor": "convert_task_to_lerobot", // Aggregation processor
    "params": {
        "source_task_dir": "src://{task_name}", // Pass full source path
        "repo_id": "lerobot/{task_name}",
        "cameras": {
            "image": "raw_video/cam_high.mp4",
            "wrist": "raw_video/cam_wrist.mp4"
        }
    }
}
```

### Scenario 5: Instruction Extraction & Cleaning
**Goal**: Source metadata is a complex JSON; extract only the first instruction string to a TXT file.

**Config**:
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

## 5. Debugging Tips

If your conversion results in **0 jobs generated**, it usually means `hierarchy` failed to match the source directory structure.

1.  **Check Patterns**: Verify glob patterns (e.g., `episode_*` vs `episode*`).
2.  **Check Depth**: Ensure your config matches the actual depth (e.g., 3 levels vs 2 levels).
3.  **Prefer Directories**: Use `type: directory` with `context_key` whenever possible. It's more robust than `filename_match` for structured datasets.
