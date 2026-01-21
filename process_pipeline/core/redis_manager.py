import redis
import json
import time
import hashlib
from typing import List, Dict, Optional
from .context import EpisodeContext
from .config import RedisConfig

class RedisManager:
    def __init__(self, config: RedisConfig, dataset_id: str = None, verbose: bool = True):
        self.enabled = config.enabled
        self.verbose = verbose
        # If dataset_id is not provided (e.g. in workers), we might need to rely on what's passed
        # However, runners should usually init with the id.
        # If none, we fallback to a default or require it.
        # For backward compatibility during migration, default to "default".
        self.dataset_id = dataset_id if dataset_id else "default"
        
        if self.enabled:
            self.r = redis.Redis(
                host=config.host, 
                port=config.port, 
                db=config.db, 
                password=config.password,
                decode_responses=True
            )
            # Check connection
            try:
                self.r.ping()
                if self.verbose:
                    print(f"Connected to Redis at {config.host}:{config.port}/{config.db} (Dataset ID: {self.dataset_id})")
            except redis.ConnectionError as e:
                # Always print errors
                print(f"Failed to connect to Redis: {e}")
                print("Redis functionality will be disabled.")
                self.enabled = False

    def _get_key(self, task_name: str, episode_name: str) -> str:
        # Namespace keys by dataset_id
        return f"ep:{self.dataset_id}:{task_name}:{episode_name}"

    def register_episodes(self, episodes: List[EpisodeContext]):
        """Register episodes in Redis if they don't exist."""
        if not self.enabled:
            return

        pipe = self.r.pipeline()
        count = 0
        set_key = f"dataset:{self.dataset_id}:episodes"
        
        for ep in episodes:
            key = self._get_key(ep.task_name, ep.episode_name)
            if not self.r.exists(key):
                pipe.hset(key, mapping={
                    "created_at": time.time(),
                    "global_valid": "unknown"
                })
                pipe.sadd(set_key, key)
                count += 1
        
        pipe.execute()
        if count > 0:
            print(f"Registered {count} new episodes in Redis.")

    def get_episode_status(self, task_name: str, episode_name: str, step_name: str) -> str:
        if not self.enabled:
            return "pending"
        
        key = self._get_key(task_name, episode_name)
        status = self.r.hget(key, f"step_{step_name}")
        return status if status else "pending"

    def is_global_invalid(self, task_name: str, episode_name: str) -> bool:
        if not self.enabled:
            return False
        key = self._get_key(task_name, episode_name)
        val = self.r.hget(key, "global_valid")
        return val == "false"

    def update_step_status(self, task_name: str, episode_name: str, step_name: str, status: str, message: str = None):
        if not self.enabled:
            return

        key = self._get_key(task_name, episode_name)
        mapping = {
            f"step_{step_name}": status,
            "last_updated": time.time()
        }
        if message:
            mapping[f"step_{step_name}_msg"] = message
            
        # Special handling for validation step
        if step_name == "validation":
            if status == "failed":
                mapping["global_valid"] = "false"
            elif status == "success":
                mapping["global_valid"] = "true"

        self.r.hset(key, mapping=mapping)

    def mark_processing(self, task_name: str, episode_name: str, step_name: str) -> bool:
        if not self.enabled:
            return True

        key = self._get_key(task_name, episode_name)
        current = self.r.hget(key, f"step_{step_name}")
        if current in ["processing", "success"]:
            return False
            
        self.r.hset(key, f"step_{step_name}", "processing")
        return True

    def filter_pending_episodes(self, episodes: List[EpisodeContext], step_name: str) -> List[EpisodeContext]:
        if not self.enabled:
            return episodes

        pending = []
        skipped_count = 0
        invalid_count = 0
        completed_count = 0

        pipe = self.r.pipeline()
        for ep in episodes:
            key = self._get_key(ep.task_name, ep.episode_name)
            pipe.hmget(key, [f"step_{step_name}", "global_valid"])
        
        results = pipe.execute()
        
        for ep, (step_status, global_valid) in zip(episodes, results):
            if global_valid == "false":
                invalid_count += 1
                continue
                
            if step_status == "success":
                completed_count += 1
                ep.status = "skipped"
                ep.reason = "Already completed (Redis)"
                continue
                
            pending.append(ep)

        print(f"Redis Filter [{step_name}]: {len(pending)} pending (including retries), {completed_count} completed, {invalid_count} invalid.")
        return pending

    def clear_all_status(self):
        """Delete all keys related to THIS dataset."""
        if not self.enabled:
            return
            
        set_key = f"dataset:{self.dataset_id}:episodes"
        keys = self.r.smembers(set_key)
        
        if keys:
            # Delete all episode keys
            # Split into chunks to avoid blocking redis if too many
            keys_list = list(keys)
            batch_size = 1000
            for i in range(0, len(keys_list), batch_size):
                batch = keys_list[i:i+batch_size]
                self.r.delete(*batch)
            
            # Delete the set key itself
            self.r.delete(set_key)
            print(f"Cleared {len(keys)} episode records for dataset {self.dataset_id} from Redis.")
        else:
            print(f"No episode records found in Redis for dataset {self.dataset_id} to clear.")

    def check_all_steps_completed(self, episodes: List[EpisodeContext], steps: List[str]) -> bool:
        if not self.enabled:
            return False
            
        pipe = self.r.pipeline()
        for ep in episodes:
            key = self._get_key(ep.task_name, ep.episode_name)
            pipe.hmget(key, [f"step_{s}" for s in steps] + ["global_valid"])
            
        results = pipe.execute()
        
        for res in results:
            global_valid = res[-1]
            if global_valid == "false":
                continue 
                
            step_statuses = res[:-1]
            for s in step_statuses:
                if s != "success" and s != "skipped":
                    return False
        return True
