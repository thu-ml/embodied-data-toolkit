import numpy as np
import torch

try:
    import pyarrow.parquet as pq
    import pandas as pd
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

def read_parquet_to_tensor(parquet_path, column_key=None, fuzzy_search_key="action"):
    """
    Reads a specific column from a parquet file and converts it to a PyTorch tensor.
    
    Args:
        parquet_path (str or Path): Path to the parquet file.
        column_key (str, optional): Exact column name to read.
        fuzzy_search_key (str, optional): Substring to search for if exact key is not found (e.g. 'action').
        
    Returns:
        torch.Tensor: The data as a float tensor, or None if column not found/error.
    """
    if not HAS_PYARROW:
        raise ImportError("pyarrow and pandas are required for parquet operations.")
        
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        target_col = None
        if column_key and column_key in df.columns:
            target_col = column_key
        elif fuzzy_search_key:
            for c in df.columns:
                if fuzzy_search_key in c:
                    target_col = c
                    break
        
        if not target_col:
            return None
            
        # Extract data
        # Handle cases where data is list of arrays or flattened
        sample = df[target_col].iloc[0]
        if isinstance(sample, (np.ndarray, list)):
             data = np.stack(df[target_col].to_numpy())
        else:
             data = df[target_col].to_numpy()
             
        return torch.from_numpy(data).float()
        
    except Exception as e:
        print(f"Error reading parquet {parquet_path}: {e}")
        return None

