import argparse
import sys
import os
import atexit
import multiprocessing as mp
from pathlib import Path

# Add current dir to path to ensure imports work if run from outside
sys.path.append(str(Path(__file__).parent))

from core.config import PipelineConfig
from core.pipeline import Pipeline

# Import processors to ensure they are registered
import processors


def restore_terminal():
    """Force restore terminal echo on exit."""
    os.system('stty echo 2>/dev/null')

def main():
    # Set start method to spawn for CUDA compatibility in multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Register cleanup handler
    atexit.register(restore_terminal)

    parser = argparse.ArgumentParser(description="Unified Robot Data Processing Pipeline (Layer-wise)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    # Optional overrides
    parser.add_argument("--src", help="Override source dataset root")
    parser.add_argument("--dest", help="Override destination dataset root")
    
    args = parser.parse_args()
    
    # Load Config
    try:
        pipeline_cfg = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # Apply overrides
    if args.src:
        pipeline_cfg.global_cfg.src = args.src
    if args.dest:
        pipeline_cfg.global_cfg.dest = args.dest
        
    # Check mode
        
    mode = pipeline_cfg.global_cfg.mode
        
    if mode != "clear":
        print(f"Configuration loaded from {args.config}")
    
    # Run Pipeline
    try:
        pipeline = Pipeline(pipeline_cfg)
        
        if mode == "clear":
            pipeline.clear_all_data()
        elif mode == "process":
            pipeline.run()
        else:
            print(f"Unknown mode: {mode}. Please use 'process' or 'clear'.")
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure terminal is restored even if atexit misses it (e.g. some signals)
        restore_terminal()

if __name__ == "__main__":
    main()
