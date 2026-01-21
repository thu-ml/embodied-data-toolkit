import os
import json
import hashlib
import time
import concurrent.futures
from typing import List, Dict, Set
from .planner import Job
from .registry import ProcessorRegistry

STATE_FILE_NAME = ".conversion_state.json"

def compute_job_hash(job: Job) -> str:
    """
    Computes a deterministic hash for the job configuration.
    Includes: Processor Name, Params, Target Path.
    """
    # Sort params to ensure deterministic order
    try:
        params_str = json.dumps(job.params, sort_keys=True)
    except TypeError:
        # Fallback for non-serializable params (e.g. sets)
        params_str = str(job.params)
        
    data = f"{job.processor_name}|{job.target_path}|{params_str}"
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def worker_execute_job(job: Job):
    """
    Worker function to execute a single job.
    """
    try:
        func = ProcessorRegistry.get(job.processor_name)
        if not func:
            raise ValueError(f"Processor '{job.processor_name}' not found in registry.")
        
        # Ensure target dir exists
        target_dir = os.path.dirname(job.target_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        print(f"Processing: {os.path.basename(job.target_path)} ({job.processor_name})")
        func(job.target_path, job.params, job.context)
        return True, None
    except Exception as e:
        import traceback
        return False, f"{str(e)}\n{traceback.format_exc()}"

class JobExecutor:
    def __init__(self, dest_root: str, max_workers: int = 4, resume: bool = True):
        self.dest_root = dest_root
        self.max_workers = max_workers
        self.resume = resume
        self.state_path = os.path.join(dest_root, STATE_FILE_NAME)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if not self.resume or not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_state(self):
        if not self.resume:
            return
        # Atomic write
        temp_path = self.state_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(self.state, f, indent=2)
        os.replace(temp_path, self.state_path)

    def execute(self, jobs: List[Job]):
        # 1. Filter jobs
        jobs_map = {j.id: j for j in jobs}
        pending_ids = set(jobs_map.keys())
        completed_ids = set()
        
        # Check cache
        to_skip = set()
        for job in jobs:
            job_hash = compute_job_hash(job)
            if self.resume and self.state.get(job.target_path) == job_hash and os.path.exists(job.target_path):
                to_skip.add(job.id)
                # print(f"Skipping cached: {os.path.basename(job.target_path)}")

        # Mark skipped as completed immediately so dependents can run
        completed_ids.update(to_skip)
        pending_ids.difference_update(to_skip)

        print(f"Total Jobs: {len(jobs)}, Cached: {len(to_skip)}, To Run: {len(pending_ids)}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job_id = {}
            
            while pending_ids or future_to_job_id:
                # 2. Submit ready jobs
                # A job is ready if all its dependencies are in completed_ids
                # AND it is in pending_ids
                
                ready_to_submit = []
                for jid in list(pending_ids):
                    job = jobs_map[jid]
                    dependencies_met = all(dep in completed_ids for dep in job.dependencies)
                    
                    if dependencies_met:
                        ready_to_submit.append(jid)
                
                for jid in ready_to_submit:
                    job = jobs_map[jid]
                    future = executor.submit(worker_execute_job, job)
                    future_to_job_id[future] = jid
                    pending_ids.remove(jid)
                
                if not future_to_job_id and not pending_ids:
                    break
                    
                # 3. Wait for at least one result
                if future_to_job_id:
                    done, _ = concurrent.futures.wait(
                        future_to_job_id.keys(), 
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        jid = future_to_job_id.pop(future)
                        job = jobs_map[jid]
                        
                        try:
                            success, error_msg = future.result()
                            if success:
                                completed_ids.add(jid)
                                # Update state
                                job_hash = compute_job_hash(job)
                                self.state[job.target_path] = job_hash
                            else:
                                print(f"Job failed: {job.target_path}\nError: {error_msg}")
                                # Strategy: Fail hard? Or continue independent branches?
                                # For now, we don't add to completed_ids, so dependents won't run.
                                pass 
                        except Exception as e:
                            print(f"Executor exception for {job.target_path}: {e}")

                    # Periodic save (or save after every batch)
                    self._save_state()
                
                # Small sleep to prevent tight loop if no jobs ready but futures running
                # (Actually wait() blocks, so this is fine)

        # Final save
        self._save_state()
        print("Execution finished.")

