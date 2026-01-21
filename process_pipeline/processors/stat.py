import json
import time
import numpy as np
from pathlib import Path
from core.runners import DatasetRunner
from core.registry import ProcessorRegistry
from multiprocessing import Pool
from utils.stat_utils import process_file_chunk

@ProcessorRegistry.register("stat")
class StatProcessor(DatasetRunner):
    """
    Dataset-level processor that computes global statistics (min, max, mean, std)
    for qpos/action dimensions across the entire dataset.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.output_path = self.config.get("output_path", "dataset_stats.json")
        self.dataset_name = self.config.get("dataset_name", "default_dataset")
        # Reuse global workers config if not specified
        self.num_workers = self.workers 

    def process_global(self, dataset_context):
        dest_root = Path(dataset_context.episodes[0].dest_episode_dir).parent.parent
        # Or use config dest if available, but dataset_context is safer for current state
        # Actually, self.config might have 'dest' if passed correctly, but let's scan the context or configured dest.
        # Since this is a post-processing step, we scan the DESTINATION directory where processed qpos.pt reside.
        
        # We can also iterate over dataset_context.episodes to get paths directly
        # This is more robust than scanning directories again.
        pt_files = []
        for ep in dataset_context.episodes:
            qpos_path = ep.dest_episode_dir / "qpos.pt"
            if qpos_path.exists():
                pt_files.append(qpos_path)
        
        total_files = len(pt_files)
        if total_files == 0:
            print(f"[{self.step_name}] No qpos.pt files found to compute statistics.")
            return dataset_context
            
        print(f"[{self.step_name}] Computing statistics for {total_files} files using {self.num_workers} workers...")
        
        t0 = time.time()
        
        # Split files into chunks for multiprocessing
        chunk_size = max(1, total_files // self.num_workers)
        chunks = [pt_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
        
        global_mins = []
        global_maxs = []
        global_sum = None
        global_sq_sum = None
        total_samples = 0
        
        with Pool(self.num_workers) as pool:
            results = pool.map(process_file_chunk, chunks)
            
        # Aggregate results
        for res in results:
            if res is None: continue
            
            # Mins/Maxs are lists of arrays
            if res["mins"]:
                global_mins.append(np.min(np.vstack(res["mins"]), axis=0))
            if res["maxs"]:
                global_maxs.append(np.max(np.vstack(res["maxs"]), axis=0))
            
            # Sums for mean/std
            for s, sq, c in zip(res["sums"], res["sq_sums"], res["counts"]):
                if global_sum is None:
                    global_sum = np.zeros_like(s)
                    global_sq_sum = np.zeros_like(sq)
                global_sum += s
                global_sq_sum += sq
                total_samples += c

        if total_samples == 0:
             print(f"[{self.step_name}] No valid data samples found.")
             return dataset_context

        # Final computation
        final_min = np.min(np.vstack(global_mins), axis=0)
        final_max = np.max(np.vstack(global_maxs), axis=0)
        final_mean = global_sum / total_samples
        
        # Variance = E[X^2] - (E[X])^2
        # Std = sqrt(Variance)
        mean_sq = global_sq_sum / total_samples
        final_std = np.sqrt(np.maximum(mean_sq - final_mean**2, 0)) # maximum to avoid negative due to precision

        elapsed = time.time() - t0
        
        # Save results
        stats_data = {
            self.dataset_name: {
                'min': final_min.tolist(),
                'max': final_max.tolist(),
                'avg': final_mean.tolist(),
                'std': final_std.tolist(),
                'file_count': total_files,
                'total_frames': int(total_samples),
                'action_dim': int(final_min.shape[0]),
                'processing_time_seconds': elapsed
            }
        }
        
        # Determine output path. If relative, save to dataset root.
        out_p = Path(self.output_path)
        if not out_p.is_absolute():
            # Save to dataset destination root
             out_p = dest_root / self.output_path
             
        # Merge with existing if needed (optional, here we overwrite for simplicity or read-update)
        if out_p.exists():
             try:
                 with open(out_p, 'r') as f:
                     existing = json.load(f)
                     existing.update(stats_data)
                     stats_data = existing
             except:
                 pass

        with open(out_p, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=4)
            
        print(f"[{self.step_name}] Statistics saved to {out_p}")
        print(f"[{self.step_name}] Processed {total_files} files ({total_samples} frames) in {elapsed:.2f}s")

        return dataset_context
