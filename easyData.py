"""
EasyData: Utilities for easy data loading and manipulation.

This module provides convenient functions for loading and processing data files,
particularly JSON files, into pandas DataFrames.

@author Jake Bohman
@date 7/23/2024
"""

import pandas as pd
import glob
import os
from typing import List, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Public API - functions available when using "from easyData import *"
__all__ = [
    'load_jsons',
    'load_csvs'
]

def load_jsons(path: str, lines: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Load all JSON files from a given path pattern into a single DataFrame.
    
    Args:
        path (str): File path pattern (supports wildcards like '*.json' or '/path/to/files/*.json')
        lines (bool, optional): Whether to read JSON lines format. Defaults to True.
        verbose (bool, optional): Whether to print progress information. Defaults to True.
    
    Returns:
        pd.DataFrame: Combined DataFrame from all JSON files
        
    Raises:
        FileNotFoundError: If no JSON files are found matching the pattern
        ValueError: If no valid JSON data could be loaded
        
    Example:
        >>> df = load_jsons('/path/to/data/*.json')
        >>> df = load_jsons('/path/to/data/file.json', lines=False)
    """
    if not isinstance(path, str):
        raise TypeError("Path must be a string")
    
    files = glob.glob(path)
    json_files = [f for f in files if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found matching pattern: {path}")
    
    if verbose:
        logger.info(f"Found {len(json_files)} JSON files to process")
    
    dfs = []
    failed_files = []
    
    for file_path in json_files:
        try:
            if verbose:
                logger.info(f"Loading: {os.path.basename(file_path)}")
            
            temp_data = pd.read_json(file_path, lines=lines)
            
            if not temp_data.empty:
                # Add source file column for traceability
                temp_data['_source_file'] = os.path.basename(file_path)
                dfs.append(temp_data)
            else:
                logger.warning(f"Empty data in file: {file_path}")
                
        except (ValueError, pd.errors.JSONDecodeError) as e:
            logger.error(f"Failed to read {file_path}: {e}")
            failed_files.append(file_path)
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            failed_files.append(file_path)
    
    if not dfs:
        raise ValueError("No valid JSON data could be loaded from any files")
    
    # Combine all DataFrames
    combined_data = pd.concat(dfs, ignore_index=True, sort=False)
    
    if verbose:
        logger.info(f"Successfully loaded {len(dfs)} files with {len(combined_data)} total records")
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    return combined_data


def load_csvs(path: str, **kwargs) -> pd.DataFrame:
    """
    Load all CSV files from a given path pattern into a single DataFrame.
    
    Args:
        path (str): File path pattern (supports wildcards like '*.csv')
        **kwargs: Additional arguments passed to pd.read_csv()
    
    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files
        
    Raises:
        FileNotFoundError: If no CSV files are found matching the pattern
        ValueError: If no valid CSV data could be loaded
    """
    if not isinstance(path, str):
        raise TypeError("Path must be a string")
    
    files = glob.glob(path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {path}")
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    dfs = []
    failed_files = []
    
    for file_path in csv_files:
        try:
            logger.info(f"Loading: {os.path.basename(file_path)}")
            temp_data = pd.read_csv(file_path, **kwargs)
            
            if not temp_data.empty:
                temp_data['_source_file'] = os.path.basename(file_path)
                dfs.append(temp_data)
            else:
                logger.warning(f"Empty data in file: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            failed_files.append(file_path)
    
    if not dfs:
        raise ValueError("No valid CSV data could be loaded from any files")
    
    combined_data = pd.concat(dfs, ignore_index=True, sort=False)
    
    logger.info(f"Successfully loaded {len(dfs)} files with {len(combined_data)} total records")
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    return combined_data
