import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import re
from datetime import datetime

class BinanceDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/Users/igyuho/Desktop/LOB-data/processed_data",
        window_size: int = 100,
        prediction_horizon: int = 120,  # Additional timesteps for prediction (60s ÷ 0.5s = 120 steps)
        n_levels: int = 100,
        threshold: float = 0.0005  # 0.05% price change threshold
    ):
        """
        Initialize Binance Dataset
        
        Args:
            data_dir: Root directory containing 'depth' and 'trades' folders
            window_size: Number of orderbook states in each sample
            prediction_horizon: Number of additional timesteps to look ahead for prediction
            n_levels: Number of price levels to use (default: 100)
            threshold: Price change threshold for label generation
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.total_window = window_size + prediction_horizon  # Total required consecutive timestamps
        self.n_levels = n_levels
        self.threshold = threshold
        
        # Load file information
        self.depth_files = self._get_sorted_files('depth')
        self.trades_files = self._get_sorted_files('trades')
        
        # Create index mapping for lazy loading
        self.file_indices = self._create_file_indices()
        print(f"======= the total number of chunks are {len(self.file_indices)} =======")

        # Cache for current file data
        self.current_depth_file = None
        self.current_trades_file = None
        self.current_depth_data = None
        self.current_trades_data = None

    def _get_sorted_files(self, folder: str) -> List[Path]:
        """Get sorted list of parquet files from specified folder"""
        files = list(self.data_dir.joinpath(folder).glob('*.parquet'))
        
        def extract_timestamp(filename: str) -> datetime:
            # Extract first timestamp from filename
            match = re.search(r'(\d{8}_\d{6})', str(filename))
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
            return datetime.min
        
        return sorted(files, key=lambda x: extract_timestamp(str(x)))

    def _create_file_indices(self) -> List[Dict]:
        """Create mapping of indices to file locations considering timestamp continuity"""
        indices = []
        current_idx = 0
        all_timestamps = []
        file_boundaries = []  # Store file boundaries and their corresponding files
        
        # First pass: collect all timestamps and file boundaries
        for depth_file, trades_file in zip(self.depth_files, self.trades_files):
            depth_df = pd.read_parquet(depth_file)
            if len(depth_df) == 0:
                continue
            
            start_idx = len(all_timestamps)
            all_timestamps.extend(depth_df['timestamp'].tolist())
            file_boundaries.append({
                'start_idx': start_idx,
                'end_idx': len(all_timestamps),
                'depth_file': depth_file,
                'trades_file': trades_file
            })
        
        # Convert to numpy array for faster operations
        timestamps = np.array(all_timestamps)
        
        # Find gaps larger than 10000ms
        time_diffs = np.diff(timestamps)
        gap_indices = np.where(time_diffs > pd.Timedelta(milliseconds=10000))[0]
        
        # Create valid ranges excluding gaps
        valid_ranges = []
        start = 0
        
        for gap_idx in gap_indices:
            if gap_idx - start >= self.total_window:  # Check against total window size
                valid_ranges.append((start, gap_idx + 1))
            start = gap_idx + 1
        
        # Add the last range if valid
        if len(timestamps) - start >= self.total_window:
            valid_ranges.append((start, len(timestamps)))
        
        # Create file indices mapping considering valid ranges
        for valid_start, valid_end in valid_ranges:
            # Find which files this range spans
            span_files = []
            for boundary in file_boundaries:
                if (valid_start < boundary['end_idx'] and 
                    valid_end > boundary['start_idx']):
                    span_files.append(boundary)
            
            if not span_files:
                continue
            
            # Calculate number of valid samples in this range
            n_samples = valid_end - valid_start - self.total_window + 1
            
            if n_samples > 0:
                indices.append({
                    'global_start_idx': current_idx,
                    'global_end_idx': current_idx + n_samples,
                    'local_start_idx': valid_start,
                    'local_end_idx': valid_end,
                    'files': span_files
                })
                current_idx += n_samples
        
        return indices

    def _load_file_data(self, depth_file: Path, trades_file: Path) -> None:
        """Load data from files into memory"""
        self.current_depth_data = pd.read_parquet(depth_file)
        self.current_trades_data = pd.read_parquet(trades_file)
        self.current_depth_file = depth_file
        self.current_trades_file = trades_file
        
        # Ensure timestamps are pandas Timestamp objects
        if not isinstance(self.current_depth_data['timestamp'].iloc[0], pd.Timestamp):
            self.current_depth_data['timestamp'] = pd.to_datetime(
                self.current_depth_data['timestamp']
            )
        if not isinstance(self.current_trades_data['timestamp'].iloc[0], pd.Timestamp):
            self.current_trades_data['timestamp'] = pd.to_datetime(
                self.current_trades_data['timestamp']
            )

    def _get_file_for_index(self, idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """Get the appropriate file and local index for the given global index"""
        for index_range in self.file_indices:
            if (index_range['global_start_idx'] <= idx < 
                index_range['global_end_idx']):
                
                local_idx = (index_range['local_start_idx'] + 
                            (idx - index_range['global_start_idx']))
                
                # Find which file contains this index
                target_file = None
                for file_info in index_range['files']:
                    if (file_info['start_idx'] <= local_idx < 
                        file_info['end_idx']):
                        target_file = file_info
                        break
                
                if target_file is None:
                    raise IndexError("Invalid index mapping")
                
                # Load file if needed
                if target_file['depth_file'] != self.current_depth_file:
                    self._load_file_data(
                        target_file['depth_file'],
                        target_file['trades_file']
                    )
                
                return (
                    self.current_depth_data,
                    self.current_trades_data,
                    local_idx - target_file['start_idx']
                )
        
        raise IndexError("Index out of bounds")

    def _get_label(self, window_data: pd.DataFrame) -> int:
        """
        Generate label based on price movement in prediction horizon
        
        Args:
            window_data: DataFrame containing both input window and prediction horizon data
        """
        # Split window into input and prediction parts
        input_data = window_data.iloc[:self.window_size]
        prediction_data = window_data.iloc[self.window_size:]
        
        # Get last trade price from input window
        last_trade = self.current_trades_data[
            self.current_trades_data['timestamp'] <= input_data.iloc[-1]['timestamp']
        ].iloc[-1]
        
        # Get trades in prediction window
        relevant_trades = self.current_trades_data[
            (self.current_trades_data['timestamp'] > input_data.iloc[-1]['timestamp']) &
            (self.current_trades_data['timestamp'] <= prediction_data.iloc[-1]['timestamp'])
        ]
        
        if len(relevant_trades) == 0:
            return 0  # No price change
            
        # Calculate volume-weighted average price (VWAP) for prediction window
        vwap = (relevant_trades['price'] * relevant_trades['quantity']).sum() / relevant_trades['quantity'].sum()
        
        # Calculate returns
        returns = (vwap - last_trade['price']) / last_trade['price']
        
        # Generate label
        if returns > self.threshold:
            return 1    # Price up
        elif returns < -self.threshold:
            return -1   # Price down
        return 0        # Price stable

    def __len__(self) -> int:
        """Total number of samples across all files"""
        if not self.file_indices:
            return 0
        # Get the last dictionary's global_end_idx
        return self.file_indices[-1]['global_end_idx']

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        # Get appropriate file and local index
        depth_data, trades_data, local_idx = self._get_file_for_index(idx)
        
        # Extract window of orderbook data including prediction horizon
        window_data = depth_data.iloc[local_idx:local_idx + self.total_window]
        
        # Get input window features (first window_size rows)
        input_window = window_data.iloc[:self.window_size]
        
        # Create feature tensor by selecting columns based on level
        feature_columns = []
        for i in range(1, self.n_levels + 1):
            feature_columns.extend([
                f'ask_price_{i}',
                f'ask_volume_{i}',
                f'bid_price_{i}',
                f'bid_volume_{i}'
            ])
        
        features = input_window[feature_columns].values
        
        # Generate label using full window
        label = self._get_label(window_data)
        
        return torch.FloatTensor(features).unsqueeze(0), label

def create_binance_dataloader(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 100,
    prediction_horizon: int = 120,  # 60 seconds with 0.5s intervals
    n_levels: int = 100,
    threshold: float = 0.0005,
    shuffle: bool = True,
    num_workers: int = 0  # Changed to 0 by default
) -> DataLoader:
    """Create a DataLoader for the Binance dataset"""
    dataset = BinanceDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        n_levels=n_levels,
        threshold=threshold
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context='fork' if num_workers > 0 else None
    )
