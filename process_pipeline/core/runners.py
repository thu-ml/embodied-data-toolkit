from abc import ABC, abstractmethod
from typing import List, Dict, Any
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from collections import defaultdict

from .interface import BaseProcessor
from .context import EpisodeContext, DatasetContext
from .status import EpisodeStatusManager
from .redis_manager import RedisManager
from .config import RedisConfig


class EpisodeRunner(BaseProcessor):
    """
    Standard parallel processing at the episode level.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.redis_cfg = self.config.get("redis_cfg")
        # Extract dataset_id passed from Pipeline
        self.redis_dataset_id = self.config.get("redis_dataset_id")
        self.enable_local_status = self.config.get("enable_local_status", True)
        self._init_managers()

    def _init_managers(self):
        """Helper to initialize non-pickleable resources."""
        self.redis_manager = None
        if self.redis_cfg:
            self.redis_manager = RedisManager(self.redis_cfg, dataset_id=self.redis_dataset_id, verbose=False)

    def __getstate__(self):
        """Pickle hook: Drop non-pickleable resources."""
        state = self.__dict__.copy()
        if 'redis_manager' in state:
            del state['redis_manager']
        return state

    def __setstate__(self, state):
        """Unpickle hook: Re-initialize resources in subprocess."""
        self.__dict__.update(state)
        self._init_managers()

    def process_dataset(self, dataset_context: DatasetContext) -> DatasetContext:
        episodes = dataset_context.episodes
        
        todo_episodes = []
        skipped_episodes = []
        force_rerun = self.config.get("overwrite", False)
        
        # 1. Determine which episodes NEED processing vs which are ALREADY done/skipped
        for ep in episodes:
            # Check if force rerun is enabled
            if force_rerun:
                todo_episodes.append(ep)
                continue

            # Check local status file if enabled
            current_status = None
            if self.enable_local_status:
                status_mgr = EpisodeStatusManager(ep.dest_episode_dir, enabled=True)
                current_status = status_mgr.get_step_status(self.step_name)
            
            # If status file says "success", skip without re-processing
            if current_status == "success":
                ep.status = "skipped"
                ep.reason = "Already completed (status file)"
                skipped_episodes.append(ep)
                continue
            
            # For failed/skipped/None status, check if data actually exists via check_completed()
            # If check_completed returns True, we can mark it as success (data is valid)
            if self.check_completed(ep):
                ep.status = "skipped"
                ep.reason = "Already completed (verified by check_completed)"
                skipped_episodes.append(ep)
                
                # Optional: Backfill status if missing but data exists?
                # This ensures .status.json reflects reality even if we skip execution.
                if self.enable_local_status and current_status != "success":
                     # We can't easily write to file here in main process if we want to be safe/clean,
                     # but EpisodeStatusManager handles file locking (or json atomic write).
                     # Let's do it to solve the user's confusion.
                     try:
                         status_mgr = EpisodeStatusManager(ep.dest_episode_dir, enabled=True)
                         status_mgr.update_step(self.step_name, "success", message="Skipped (Verified existing data)")
                         # Also update redis?
                         if self.redis_manager and self.redis_manager.enabled:
                             self.redis_manager.update_step_status(
                                 ep.task_name, ep.episode_name, self.step_name, "success", message="Skipped (Verified existing data)"
                             )
                     except Exception as e:
                         print(f"Warning: Failed to backfill status for {ep.episode_name}: {e}")

            else:
                # Need to process: either failed before, or data doesn't exist
                todo_episodes.append(ep)
        
        # Redis filtering (Optional optimization to prune TODO list if we trust Redis)
        # If we rely on check_completed, Redis is less critical for skipping, 
        # but useful for distributing work.
        # For simplicity in this fix, we stick to the local determination above.

        if not todo_episodes:
            print(f"[{self.step_name}] All {len(episodes)} episodes already completed or invalid. Skipped.")
            dataset_context.update_results(episodes)
            return dataset_context

        if skipped_episodes:
            print(f"[{self.step_name}] Skipping {len(skipped_episodes)} already completed/invalid episodes.")

        print(f"[{self.step_name}] Processing {len(todo_episodes)} episodes with {self.workers} workers...")
        
        results = []
        with Pool(self.workers) as pool:
            for res in tqdm(pool.imap(self._episode_wrapper, todo_episodes), total=len(todo_episodes)):
                results.append(res)
                
        # Combine results from both lists
        final_episodes = []
        
        # 1. Add skipped episodes (which are already updated in-place above)
        final_episodes.extend(skipped_episodes)
        
        # 2. Add processed episodes (from pool results)
        result_map = {ep.dest_episode_dir: ep for ep in results}
        
        # Note: todo_episodes contains the original context objects before processing.
        # We need to replace them with the processed ones from result_map.
        # But wait, dataset_context.episodes needs to be updated.
        
        # Better strategy: rebuild the full list in original order or just update the master list
        
        # Create a map for quick lookup
        processed_map = {ep.dest_episode_dir: ep for ep in results}
        skipped_map = {ep.dest_episode_dir: ep for ep in skipped_episodes}
        
        final_episodes = []
        for ep in episodes:
            if ep.dest_episode_dir in processed_map:
                final_episodes.append(processed_map[ep.dest_episode_dir])
            elif ep.dest_episode_dir in skipped_map:
                final_episodes.append(skipped_map[ep.dest_episode_dir])
            else:
                # Should not happen if logic covers all, but fallback to original
                final_episodes.append(ep)
                
        dataset_context.update_results(final_episodes)
        
        # --- ADDED LOGIC: Check for failures ---
        failed_episodes = [ep for ep in final_episodes if ep.status in ("failed", "error")]
        if failed_episodes:
            print(f"[{self.step_name}] {len(failed_episodes)} episodes FAILED.")
            self._log_failures(failed_episodes)
        # ---------------------------------------
        
        return dataset_context

    def _episode_wrapper(self, episode_context: EpisodeContext) -> EpisodeContext:
        """
        Internal wrapper to handle exceptions and status updates.
        Executed in Worker Process.
        """
        # Double check completion inside worker just in case? No, expensive.
        status_mgr = EpisodeStatusManager(episode_context.dest_episode_dir, enabled=self.enable_local_status)
        
        try:
            # Mark processing
            if self.redis_manager:
                self.redis_manager.mark_processing(episode_context.task_name, episode_context.episode_name, self.step_name)

            result_ctx = self.process_episode(episode_context)
            self._update_status(status_mgr, result_ctx)
            return result_ctx
        except Exception as e:
            episode_context.fail(f"Exception in {self.step_name}: {str(e)}")
            self._update_status(status_mgr, episode_context)
            return episode_context

    def _update_status(self, status_mgr: EpisodeStatusManager, result_ctx: EpisodeContext):
        # Determine status
        should_update_redis = True
        
        if result_ctx.status == "failed" or result_ctx.status == "error":
            status = "failed"
            msg = result_ctx.reason or result_ctx.error
        elif result_ctx.status == "skipped":
            # Runtime skip inside processor (e.g. file exists). 
            # We treat this as SUCCESS for persistence.
            status = "success" 
            msg = "Skipped (Already exists)"
        else:
            status = "success"
            msg = None

        # Update File Status (if enabled)
        status_mgr.update_step(self.step_name, status, message=msg)
        
        # Update Redis Status
        if should_update_redis and self.redis_manager and self.redis_manager.enabled:
            self.redis_manager.update_step_status(
                result_ctx.task_name, 
                result_ctx.episode_name, 
                self.step_name, 
                status, 
                message=msg
            )

    @abstractmethod
    def process_episode(self, context: EpisodeContext) -> EpisodeContext:
        pass


class TaskRunner(BaseProcessor):
    """
    Parallel processing at the task level.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.redis_cfg = self.config.get("redis_cfg")
        # Extract dataset_id
        self.redis_dataset_id = self.config.get("redis_dataset_id")
        self.enable_local_status = self.config.get("enable_local_status", True)
        self._init_managers()

    def _init_managers(self):
        self.redis_manager = None
        if self.redis_cfg:
            self.redis_manager = RedisManager(self.redis_cfg, dataset_id=self.redis_dataset_id, verbose=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'redis_manager' in state:
            del state['redis_manager']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_managers()

    def process_dataset(self, dataset_context: DatasetContext) -> DatasetContext:
        tasks = defaultdict(list)
        for ep in dataset_context.episodes:
            tasks[ep.task_name].append(ep)
            
        print(f"[{self.step_name}] Identified {len(tasks)} tasks.")
        
        task_items = []
        skipped_results = []
        force_rerun = self.config.get("overwrite", False)

        for task_name, episodes in tasks.items():
            # Check if ALL episodes in task are complete
            # If any episode needs processing, the whole task needs processing
            
            all_complete = True
            for ep in episodes:
                if force_rerun:
                    all_complete = False
                    break
                
                # Check status file
                current_status = None
                if self.enable_local_status:
                    status_mgr = EpisodeStatusManager(ep.dest_episode_dir, enabled=True)
                    current_status = status_mgr.get_step_status(self.step_name)
                
                # If status file says "success", skip this episode
                if current_status == "success":
                    continue
                
                # For failed/skipped/None status, check if data actually exists
                if not self.check_completed(ep):
                    # This episode needs processing
                    all_complete = False
                    break
            
            if all_complete and not force_rerun:
                # Mark all as skipped
                for ep in episodes:
                    ep.status = "skipped"
                    ep.reason = "Task already completed"
                    
                    # Backfill status if missing but task is effectively complete
                    # (Only trying to fix the first one or iterate all? Iterate all to be safe)
                    if self.enable_local_status:
                        try:
                            status_mgr = EpisodeStatusManager(ep.dest_episode_dir, enabled=True)
                            # Only write if not already success
                            if status_mgr.get_step_status(self.step_name) != "success":
                                status_mgr.update_step(self.step_name, "success", message="Skipped (Verified existing data)")
                                if self.redis_manager and self.redis_manager.enabled:
                                    self.redis_manager.update_step_status(
                                        ep.task_name, ep.episode_name, self.step_name, "success", message="Skipped (Verified existing data)"
                                    )
                        except Exception:
                            pass

                skipped_results.extend(episodes)
            else:
                task_items.append((task_name, episodes))

        if skipped_results:
             print(f"[{self.step_name}] Skipping {len(skipped_results)} episodes in already completed tasks.")

        print(f"[{self.step_name}] Processing {len(task_items)} tasks with {self.workers} workers...")

        results = []
        if self.workers > 1:
            with Pool(self.workers) as pool:
                for res_list in tqdm(pool.imap(self._task_wrapper, task_items), total=len(task_items)):
                    results.extend(res_list)
        else:
            for item in tqdm(task_items):
                results.extend(self._task_wrapper(item))
            
        # Combine processed results with skipped episodes
        final_episodes = []
        
        # 1. Add skipped episodes (already updated in-place)
        final_episodes.extend(skipped_results)
        
        # 2. Add processed episodes
        # Note: 'results' contains the EpisodeContext objects returned from process_task
        # Since process_task returns a list of contexts, and we extended results,
        # results is a flat list of EpisodeContexts.
        final_episodes.extend(results)
        
        # Sort or just update? Order might change but that's usually fine for DatasetContext.
        # If we want to preserve order, we can use a map strategy like EpisodeRunner.
        
        dataset_context.update_results(final_episodes)
        
        # --- ADDED LOGIC: Check for failures ---
        failed_episodes = [ep for ep in final_episodes if ep.status in ("failed", "error")]
        if failed_episodes:
            print(f"[{self.step_name}] {len(failed_episodes)} episodes failed in task processing.")
            self._log_failures(failed_episodes)
        # ---------------------------------------

        return dataset_context

    def _task_wrapper(self, args):
        task_name, episodes = args
        try:
            return self.process_task(task_name, episodes)
        except Exception as e:
            # Mark all episodes in this task as failed
            for ep in episodes:
                ep.fail(f"Exception in task {task_name}: {str(e)}")
                status_mgr = EpisodeStatusManager(ep.dest_episode_dir, enabled=self.enable_local_status)
                self._update_status(status_mgr, ep)
            return episodes

    def _update_status(self, status_mgr: EpisodeStatusManager, result_ctx: EpisodeContext):
        # Same as EpisodeRunner._update_status, simplified duplication
        should_update_redis = True
        
        if result_ctx.status == "failed" or result_ctx.status == "error":
            status = "failed"
            msg = result_ctx.reason or result_ctx.error
        elif result_ctx.status == "skipped":
            status = "success" 
            msg = "Skipped (Already exists)"
        else:
            status = "success"
            msg = None

        status_mgr.update_step(self.step_name, status, message=msg)
        
        if should_update_redis and self.redis_manager and self.redis_manager.enabled:
            self.redis_manager.update_step_status(
                result_ctx.task_name, 
                result_ctx.episode_name, 
                self.step_name, 
                status, 
                message=msg
            )

    @abstractmethod
    def process_task(self, task_name: str, episodes: List[EpisodeContext]) -> List[EpisodeContext]:
        pass


class DatasetRunner(BaseProcessor):
    """
    Global processing on the entire dataset.
    """
    def process_dataset(self, dataset_context: DatasetContext) -> DatasetContext:
        print(f"[{self.step_name}] Processing entire dataset globally...")
        try:
            return self.process_global(dataset_context)
        except Exception as e:
            print(f"[{self.step_name}] Global processing failed: {e}")
            return dataset_context

    @abstractmethod
    def process_global(self, dataset_context: DatasetContext) -> DatasetContext:
        pass


class EpisodeRunnerMixin(EpisodeRunner):
    """
    Mixin for processors that want to use EpisodeRunner logic via composition
    but need to bridge the interface.
    """
    def __init__(self, processor):
        self.processor = processor
        self.config = processor.config
        self.workers = processor.workers
        self.step_name = processor.step_name
        self.redis_cfg = self.config.get("redis_cfg")
        self.redis_dataset_id = self.config.get("redis_dataset_id")
        self.enable_local_status = self.config.get("enable_local_status", True)
        self._init_managers()
        
    def process_episode(self, context):
        return self.processor.process_episode(context)
    
    # Delegate check_completed to processor
    def check_completed(self, context):
        return self.processor.check_completed(context)

class TaskRunnerMixin(TaskRunner):
    """
    Mixin for processors that want to use TaskRunner logic via composition
    but need to bridge the interface.
    """
    def __init__(self, processor):
        self.processor = processor
        self.config = processor.config
        self.workers = processor.workers
        self.step_name = processor.step_name
        self.redis_cfg = self.config.get("redis_cfg")
        self.redis_dataset_id = self.config.get("redis_dataset_id")
        self.enable_local_status = self.config.get("enable_local_status", True)
        self._init_managers()

    def process_task(self, task_name, episodes):
        return self.processor.process_task(task_name, episodes)
    
    # Delegate check_completed to processor
    def check_completed(self, context):
        return self.processor.check_completed(context)
