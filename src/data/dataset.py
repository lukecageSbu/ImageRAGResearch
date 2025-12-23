import os
import pandas as pd
from typing import Optional

def _load_and_process_data(csv_path: str, subdir: str) -> pd.DataFrame:
    """Helper to load and process dataframes from txt files"""
    df = pd.read_csv(csv_path, dtype={'unique_ID': str})
    
    # Map 'image' to 'Path' and prepend subdir
    if 'image' in df.columns and 'Path' not in df.columns:
        df['Path'] = df['image'].apply(lambda x: os.path.join(subdir, x))
    
    # Map 'forged' to 'SPI_label' if present
    if 'forged' in df.columns and 'SPI_label' not in df.columns:
        df['SPI_label'] = df['forged']
        
    return df

def load_training_data(config) -> pd.DataFrame:
    return _load_and_process_data(config.TRAIN_DATA_PATH, 'train')

def load_test_data(config) -> pd.DataFrame:
    return _load_and_process_data(config.TEST_DATA_PATH, 'test')

def load_validation_data(config) -> pd.DataFrame:
    return _load_and_process_data(config.VALIDATION_DATA_PATH, 'val')

def get_image_path(row: pd.Series, data_dir: str) -> Optional[str]:
    image_path = row.get('Path')
    if not image_path or pd.isna(image_path):
        # Fallback to 'image' column if Path is missing (should be handled by loader though)
        image_path = row.get('image')
        
    if not image_path or pd.isna(image_path):
        return None
        
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(data_dir, image_path)

