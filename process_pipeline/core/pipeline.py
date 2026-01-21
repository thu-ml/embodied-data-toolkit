import re
import traceback
import hashlib
import shutil
from pathlib import Path
from typing import List
import time
from .context import EpisodeContext, DatasetContext
from .config import PipelineConfig
from .registry import ProcessorRegistry
from .redis_manager import RedisManager
from utils.file_utils import find_task_directories, ensure_dir


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Generate dataset_id based on source path
        src_path_str = str(Path(self.config.global_cfg.src).absolute())
        self.dataset_id = hashlib.md5(src_path_str.encode('utf-8')).hexdigest()[:8]
        
        self.redis_manager = RedisManager(config.redis_cfg, dataset_id=self.dataset_id)

    def run(self):
        """Execute the pipeline based on the provided configuration."""
        # 1. Initialize Context (Scan Source)
        dataset_context = self._initialize_context()
        
        # 2. Register episodes in Redis
        if self.redis_manager.enabled:
            print("Registering episodes in Redis...")
            self.redis_manager.register_episodes(dataset_context.episodes)
        
        # 3. Run Steps Layer-wise
        step_names = [s.name for s in self.config.steps]
        for step_config in self.config.steps:
            dataset_context = self._run_step(step_config, dataset_context)
            
            # --- ADDED LOGIC: Check failures and potentially abort ---
            # Check if any episode failed in this step
            # Note: Runners usually mark failed episodes with status="failed" or "error"
            failed_episodes = [ep for ep in dataset_context.episodes if ep.status in ("failed", "error")]
            if failed_episodes:
                print(f"\n[PIPELINE ABORT] Step '{step_config.name}' reported {len(failed_episodes)} failures.")
                print("Stopping execution of subsequent steps.")
                print(f"Please check the error logs (e.g., pipeline_errors.log in dataset root).")
                return # Stop the pipeline
            # ---------------------------------------------------------
        
        # 4. Cleanup Redis if all done
        if self.redis_manager.enabled:
            print("Checking for pipeline completion...")
            if self.redis_manager.check_all_steps_completed(dataset_context.episodes, step_names):
                print("All steps completed for all valid episodes. Clearing Redis data...")
                self.redis_manager.clear_all_status()
            else:
                print("Pipeline not fully complete. Retaining Redis data for breakpoint resumption.")
            
        print("\nProcessing Complete.")

    def clear_all_data(self):
        """Clear all data associated with this dataset (Redis + Dest Files)."""
        print(f"!!! WARNING: CLEAR MODE ENABLED !!!")
        print(f"Dataset ID: {self.dataset_id}")
        print(f"Source: {self.config.global_cfg.src}")
        print(f"Dest: {self.config.global_cfg.dest}")
        print("This will DELETE all Redis keys and destination files for this dataset.")
        print("Waiting 5 seconds... Ctrl+C to cancel.")
        time.sleep(5)
        
        # 1. Clear Redis
        if self.redis_manager.enabled:
            print("Clearing Redis data...")
            self.redis_manager.clear_all_status()
        
        # 2. Clear Dest Dir
        dest_path = Path(self.config.global_cfg.dest)
        if dest_path.exists():
            print(f"Removing destination directory: {dest_path}")
            shutil.rmtree(dest_path)
            # Re-create empty dir
            ensure_dir(dest_path)
        else:
            print(f"Destination directory does not exist: {dest_path}")
            
        print("Cleanup complete.")

    def _initialize_context(self) -> DatasetContext:
        src_path = Path(self.config.global_cfg.src)
        dest_path = Path(self.config.global_cfg.dest)
        base_dest = dest_path
        
        ensure_dir(base_dest)
        
        print(f"Source: {src_path}")
        print(f"Dest: {dest_path}")
        print(f"Pipeline Steps: {[s.name for s in self.config.steps]}")
        print(f"Dataset ID (for Redis): {self.dataset_id}")
        
        print(f"Scanning tasks in {src_path}...")
        task_dirs_data = find_task_directories(src_path, task_pattern="*")
        print(f"Found {len(task_dirs_data)} tasks.")
        
        all_episodes = []
        
        for task_data in task_dirs_data:
            task_name = task_data["name"]
            episodes_dict = task_data["episodes"]
            
            # Sort episodes naturally
            sorted_keys = sorted(episodes_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
            episodes_paths = [episodes_dict[k]["path"] for k in sorted_keys]
            
            dest_task_dir = base_dest / task_name
            ensure_dir(dest_task_dir)
            
            for ep_path in episodes_paths:
                dest_ep = dest_task_dir / ep_path.name
                ensure_dir(dest_ep)
                
                ctx = EpisodeContext(
                    src_episode_dir=ep_path,
                    task_name=task_name,
                    dest_episode_dir=dest_ep
                )
                all_episodes.append(ctx)
                
        print(f"Total episodes found: {len(all_episodes)}")
        return DatasetContext(episodes=all_episodes)

    def _run_step(self, step_config, dataset_context: DatasetContext) -> DatasetContext:
        step_name = step_config.name
        step_params = step_config.params
        
        print(f"\n" + "="*40)
        print(f"Running Step: {step_name}")
        print("="*40)
        
        proc_cls = ProcessorRegistry.get(step_name)
        if not proc_cls:
            print(f"Warning: Unknown processor '{step_name}', skipping.")
            return dataset_context
            
        try:
            # Add global workers config if not present in step params
            if "workers" not in step_params:
                step_params["workers"] = self.config.global_cfg.workers
            
            # Inject global local status config
            step_params["enable_local_status"] = self.config.global_cfg.enable_local_status
            
            # Inject step_name into config so BaseProcessor can use it for status tracking
            step_params["step_name"] = step_name
            
            # Inject redis config into params so processor can use it
            if self.redis_manager.enabled:
                step_params["redis_cfg"] = self.config.redis_cfg
                # IMPORTANT: Also pass dataset_id so workers use correct keys
                step_params["redis_dataset_id"] = self.dataset_id
                
            processor = proc_cls(config=step_params)
            
            # The processor takes the entire dataset context and returns an updated one
            dataset_context = processor.process_dataset(dataset_context)
            
            # Basic reporting
            failed = [ep for ep in dataset_context.episodes if ep.status == "failed" or ep.status == "error"]
            skipped = [ep for ep in dataset_context.episodes if ep.status == "skipped"]
            success = [ep for ep in dataset_context.episodes if ep.status == "success" or ep.status == "pending"]
            
            print(f"Step {step_name} summary: {len(success)} ok, {len(skipped)} skipped, {len(failed)} failed")
            
        except Exception as e:
            print(f"Critical error in step {step_name}: {e}")
            traceback.print_exc()
            
            # If critical runner error occurs (not just episode error), we should probably fail too?
            # But the dataset_context might be stale.
            # Let's rely on the outer check.
            
        return dataset_context
