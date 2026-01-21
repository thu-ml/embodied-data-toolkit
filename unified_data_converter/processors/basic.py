import os
import shutil
import json
from core.registry import ProcessorRegistry

@ProcessorRegistry.register("copy")
def process_copy(target_path: str, params: dict, context: dict):
    # keep the original source data
    source = params.get("source")
    if not source:
        raise ValueError("Parameter 'source' is required for copy processor")
    
    if os.path.isdir(source):
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(source, target_path)
    else:
        shutil.copy2(source, target_path)

@ProcessorRegistry.register("move")
def process_move(target_path: str, params: dict, context: dict):
    # delete the original source data
    source = params.get("source")
    shutil.move(source, target_path)

