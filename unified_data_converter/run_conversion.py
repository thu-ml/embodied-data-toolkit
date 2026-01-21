import os
import argparse
import json
import sys
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Core Components
from core.context import ContextScanner
from core.resolver import ParamResolver
from core.planner import Planner
from core.executor import JobExecutor
from core.registry import ProcessorRegistry

# Import Processors to trigger registration
import processors

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Unified Data Converter (New Architecture)")
    parser.add_argument("--config", type=str, required=True, help="Path to the mapping JSON config")
    parser.add_argument("--src_root", type=str, required=True, help="Root directory of source data")
    parser.add_argument("--dest_root", type=str, required=True, help="Root directory for output")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume capability (force re-run)")
    
    args = parser.parse_args()
    
    # 0. Setup
    resume = not args.no_resume
    if not os.path.exists(args.src_root):
        print(f"Error: Source root not found: {args.src_root}")
        return
        
    config = load_config(args.config)
    hierarchy = config.get("hierarchy", [])
    target_structure = config.get("target_structure", [])
    
    print(f"=== Starting Conversion ===")
    print(f"Source: {args.src_root}")
    print(f"Dest:   {args.dest_root}")
    print(f"Config: {args.config}")
    print(f"Resume: {resume}")
    
    start_time = time.time()
    
    # 1. Build Context Tree
    print("\n[1/4] Scanning Source Hierarchy...")
    scanner = ContextScanner(args.src_root)
    context_tree = scanner.scan(hierarchy)
    
    # 2. Plan Jobs
    print("\n[2/4] Planning Jobs...")
    resolver = ParamResolver(args.src_root, args.dest_root)
    planner = Planner(resolver)
    jobs = planner.plan_v2(target_structure, context_tree)
    print(f"Generated {len(jobs)} jobs.")
    
    # 3. Execute
    print("\n[3/4] Executing...")
    executor = JobExecutor(args.dest_root, max_workers=args.workers, resume=resume)
    executor.execute(jobs)
    
    duration = time.time() - start_time
    print(f"\n[4/4] Done in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
import processors.lerobot_expand
