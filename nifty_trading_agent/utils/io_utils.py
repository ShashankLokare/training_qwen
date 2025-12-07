"""
Input/Output utilities for the Nifty Trading Agent
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

def ensure_directories(directories: List[str]) -> None:
    """
    Ensure that all specified directories exist, creating them if necessary

    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_yaml_config(file_path: str) -> Dict:
    """
    Load YAML configuration file

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary containing configuration
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config if config else {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {file_path}: {e}")

def save_yaml_config(config: Dict, file_path: str) -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        file_path: Path to save YAML file
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
    except Exception as e:
        raise IOError(f"Error saving YAML file {file_path}: {e}")

def load_json_file(file_path: str) -> Union[Dict, List]:
    """
    Load JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        JSON data as dictionary or list
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file {file_path}: {e}")

def save_json_file(data: Union[Dict, List], file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file

    Args:
        data: Data to save (dict or list)
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=indent, default=str)
    except Exception as e:
        raise IOError(f"Error saving JSON file {file_path}: {e}")

def load_csv_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame containing CSV data
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading CSV file {file_path}: {e}")

def save_csv_file(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to CSV file

    Args:
        df: DataFrame to save
        file_path: Path to save CSV file
        **kwargs: Additional arguments for df.to_csv
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(file_path, **kwargs)
    except Exception as e:
        raise IOError(f"Error saving CSV file {file_path}: {e}")

def load_pickle_file(file_path: str) -> Any:
    """
    Load pickle file

    Args:
        file_path: Path to pickle file

    Returns:
        Unpickled object
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading pickle file {file_path}: {e}")

def save_pickle_file(data: Any, file_path: str) -> None:
    """
    Save data to pickle file

    Args:
        data: Data to pickle
        file_path: Path to save pickle file
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        raise IOError(f"Error saving pickle file {file_path}: {e}")

def load_excel_file(file_path: str, sheet_name: Optional[str] = None, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load Excel file into pandas DataFrame(s)

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name to load (None for all sheets)
        **kwargs: Additional arguments for pd.read_excel

    Returns:
        DataFrame or dict of DataFrames if multiple sheets
    """
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading Excel file {file_path}: {e}")

def save_excel_file(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], file_path: str, **kwargs) -> None:
    """
    Save DataFrame(s) to Excel file

    Args:
        data: DataFrame or dict of DataFrames to save
        file_path: Path to save Excel file
        **kwargs: Additional arguments for df.to_excel
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name='Sheet1', **kwargs)
            elif isinstance(data, dict):
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, **kwargs)
            else:
                raise ValueError("Data must be DataFrame or dict of DataFrames")
    except Exception as e:
        raise IOError(f"Error saving Excel file {file_path}: {e}")

def list_files_in_directory(directory: str, pattern: str = "*") -> List[str]:
    """
    List all files in a directory matching a pattern

    Args:
        directory: Directory path
        pattern: File pattern to match (default: all files)

    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    files = [str(f) for f in dir_path.glob(pattern) if f.is_file()]
    return files

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes

    Args:
        file_path: Path to file

    Returns:
        File size in bytes
    """
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        return 0

def file_exists(file_path: str) -> bool:
    """
    Check if file exists

    Args:
        file_path: Path to file

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()

def create_backup_file(file_path: str, suffix: str = ".backup") -> str:
    """
    Create a backup of a file

    Args:
        file_path: Original file path
        suffix: Backup file suffix

    Returns:
        Path to backup file
    """
    backup_path = str(Path(file_path)) + suffix

    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        raise IOError(f"Error creating backup of {file_path}: {e}")

def get_file_modification_time(file_path: str) -> Optional[float]:
    """
    Get file modification time as Unix timestamp

    Args:
        file_path: Path to file

    Returns:
        Modification time as float or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_mtime
    except FileNotFoundError:
        return None

def is_file_recent(file_path: str, max_age_seconds: int = 3600) -> bool:
    """
    Check if file was modified within the last N seconds

    Args:
        file_path: Path to file
        max_age_seconds: Maximum age in seconds (default: 1 hour)

    Returns:
        True if file is recent, False otherwise
    """
    import time
    mtime = get_file_modification_time(file_path)

    if mtime is None:
        return False

    current_time = time.time()
    return (current_time - mtime) <= max_age_seconds
