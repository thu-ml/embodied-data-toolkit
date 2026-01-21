from abc import ABC, abstractmethod
from typing import List
import datetime
from pathlib import Path
from .context import EpisodeContext, DatasetContext

class BaseProcessor(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.workers = self.config.get("workers", 8)
        self.step_name = self.config.get("step_name", self.__class__.__name__)

    @abstractmethod
    def process_dataset(self, dataset_context: DatasetContext) -> DatasetContext:
        """
        Process the entire dataset.
        Must be implemented by subclasses (or by Runner mixins).
        """
        pass

    def check_completed(self, context: EpisodeContext) -> bool:
        """
        Check if the output of this processor already exists and is valid.
        Can be used by runners or subclasses to skip processing.
        
        Default: False (Always re-run unless Runner status check skips it)
        """
        return False

    def _log_failures(self, failed_contexts: List[EpisodeContext], dataset_root: Path = None):
        """
        Log failure details to a file in the dataset root.
        """
        if not failed_contexts:
            return

        # Determine log path: try to use dest_dir from first context, else current dir
        if dataset_root is None:
            if failed_contexts and hasattr(failed_contexts[0], 'dest_episode_dir'):
                # Assuming dest_episode_dir is .../task_name/episode_id/
                # We want .../ (dataset root)
                # failed_contexts[0].dest_episode_dir.parent.parent might be task/dataset
                # A bit risky to guess. Let's try to find common root or just passed in.
                # Actually EpisodeRunner usually doesn't know global dataset root easily unless passed in.
                # But we can try to infer from the first failure.
                try:
                    # e.g. /data/dest/task_name/episode_0
                    dataset_root = failed_contexts[0].dest_episode_dir.parent.parent
                except:
                    dataset_root = Path(".")
            else:
                dataset_root = Path(".")

        log_file = dataset_root / "pipeline_errors.log"
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Step: {self.step_name}\n")
                f.write(f"Failed Count: {len(failed_contexts)}\n")
                f.write(f"{'='*50}\n")
                
                for ctx in failed_contexts:
                    error_msg = ctx.error or ctx.reason or "Unknown Error"
                    f.write(f"Task: {ctx.task_name}, Episode: {ctx.episode_name}\n")
                    f.write(f"Error: {error_msg}\n")
                    f.write("-" * 30 + "\n")
            
            print(f"[{self.step_name}] Error details logged to: {log_file}")
        except Exception as e:
            print(f"[{self.step_name}] Failed to write error log: {e}")
