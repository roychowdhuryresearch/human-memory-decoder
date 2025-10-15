"""Base classes and utilities for neural data processing."""

import numpy as np
import pandas as pd
import os
import pickle
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from scipy.stats import zscore
import glob
import re
import random
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler, RandomSampler, Sampler
from torchvision.transforms import transforms
from typing import Optional, Callable, Tuple, Union, Dict, Any, List
from src.param.param_data import *

class BaseNeuralDataset:
    """Base class for neural datasets with common functionality.
    
    This class provides shared functionality for processing both spike and LFP data
    from neural recordings. It handles data loading, preprocessing, filtering,
    and various data augmentation techniques.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the base neural dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.patient = config['patient']
        self.use_spike = config['use_spike']
        self.use_lfp = config['use_lfp']
        self.use_combined = config['use_combined']
        self.config = config
        
        # Initialize data containers
        self.lfp_data = []
        self.spike_data = []
        self.data = []
        self.lfp_channel_by_region = {}
        self.spike_channel_by_region = {}

    @staticmethod
    def _sort_filename_key(filename: str) -> List[Union[str, int]]:
        """Extract sort key from filename.
        
        Args:
            filename: Filename to process
            
        Returns:
            List of string and integer components for sorting
        """
        return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', filename)]

    def _get_sorted_files(self, path: str) -> List[str]:
        """Get sorted list of .npz files from directory.
        
        Args:
            path: Directory path to search
            
        Returns:
            Sorted list of .npz files
        """
        files = glob.glob(os.path.join(path, '*.npz'))
        return sorted(files, key=self._sort_filename_key)

    @staticmethod
    def get_localization_clusterless(patient: str, region: str) -> np.ndarray:
        """Get electrode localizations for a specific patient and brain region.
        
        Args:
            patient: Patient identifier
            region: Brain region identifier ('MTL', 'HPC', 'FC', etc.)
            
        Returns:
            Array of electrode names that match the criteria
        """
        region_list = {
            'MTL': ['HPC', 'AMY', 'PHC', 'ERC'],
            'HPC': ['HPC'],
            'AMY': ['AMY'],
            'FC': ['FC', 'CC', 'PARS']
        }.get(region, [])
        
        if not region_list:
            raise ValueError(f"Unknown region: {region}")

        file_path = f'data/localization_sheet/{patient}_localizations.xlsx'
        df = pd.read_excel(file_path)
        df = df[['electrode', 'BIPOLAR']]
        df = df.dropna(subset=['BIPOLAR'])
        df = df[df['electrode'].str.contains('micro', na=False)]
        df = df[df['BIPOLAR'].isin(region_list)]
        df['electrode'] = df['electrode'].str.replace('_micro-1', '')
        return df['electrode'].values

    @staticmethod
    def remove_noise_electrode(patient: str, region: str = 'WM') -> np.ndarray:
        """Remove noisy electrodes based on anatomical location.
        
        Args:
            patient: Patient identifier
            region: Region to exclude (e.g., 'WM' for white matter)
            
        Returns:
            Array of electrode names after removing noisy ones
        """
        file_path = f'data/localization_sheet/{patient}_localizations.xlsx'
        df = pd.read_excel(file_path)
        df = df[['electrode', 'BIPOLAR']]
        df = df.dropna(subset=['BIPOLAR'])
        df = df[df['electrode'].str.contains('micro', na=False)]
        df = df[~df['BIPOLAR'].isin([region])]
        df['electrode'] = df['electrode'].str.replace('_micro-1', '')
        return df['electrode'].values

    @staticmethod
    def get_localization_clusterless_by_region(patient: str, region: str) -> np.ndarray:
        """Get electrode localizations for a specific patient and brain region.
        
        Args:
            patient: Patient identifier
            region: Brain region identifier ('MTL', 'HPC', 'FC', etc.)
            
        Returns:
            Array of electrode names that match the criteria
        """
        region_list = [region]

        file_path = f'data/localization_sheet/{patient}_localizations.xlsx'
        df = pd.read_excel(file_path)
        df = df[['electrode', 'BIPOLAR']]
        df = df.dropna(subset=['BIPOLAR'])
        df = df[df['electrode'].str.contains('micro', na=False)]
        df = df[df['BIPOLAR'].isin(region_list)]
        df['electrode'] = df['electrode'].str.replace('_micro-1', '')
        return df['electrode'].values
    
    def _filter_by_region_mode(self, items: List[str], regions: List[str], mode: str = 'include') -> List[str]:
        """Filter items based on region names and filtering mode.
        
        Args:
            items: List of items to filter (can be regions or file paths)
            regions: List of region names to filter by
            mode: Filtering mode - 'include' to keep matching items, 
                 'exclude' to remove matching items
            
        Returns:
            Filtered list of items
        """
        # First attach numbers 1-8 to each region
        regions = [f"{item}{i}" for item in regions for i in range(1, 9)]
        
        # Then filter based on mode
        if mode == 'include':
            return [item for item in items if any(sub in item for sub in regions)]
        else:  # mode == 'exclude'
            return [item for item in items if not any(sub in item for sub in regions)]

    def _get_filtered_regions(self, config: Dict[str, Any]) -> List[str]:
        """Get filtered list of regions based on configuration.
        
        Args:
            config: Configuration dictionary containing patient and lesion info
            
        Returns:
            List of filtered region names
        """
        # First remove noise electrodes (WM regions)
        regions = self.remove_noise_electrode(self.patient, 'WM')
        regions = [f"{item}{i}" for item in regions for i in range(1, 9)]
        
        # Then apply lesion filtering if specified
        if config.get('lesion') != 'Full':
            sub_regions = self.get_localization_clusterless(self.patient, config['lesion'])
            mode = 'include' if config.get('lesion_mode') == 'only' else 'exclude'
            regions = self._filter_by_region_mode(regions, sub_regions, mode)
            
        return regions

    @staticmethod
    def anc(data: np.ndarray) -> np.ndarray:
        """Apply active noise cancellation to the data.
        
        Args:
            data: Input data with shape (samples, features, channels, bins)
            
        Returns:
            Noise-cancelled data with same shape as input
        """
        num_sample, _, num_channel, num_bin = data.shape
        array = data[:, 0, :, :].transpose(1, 0, 2).reshape(num_channel, -1)
        toy_array = np.copy(array)
        toy_array[toy_array > 0.2] = 0
        non_zero_counts = np.count_nonzero(toy_array, axis=0)
        thresh = 8
        noisy_indices = np.where(non_zero_counts > thresh)[0]
        subset = array[:, noisy_indices]
        subset[subset <= 0.2] = 0
        array[:, noisy_indices] = subset
        
        reshaped_back = array.reshape(num_channel, num_sample, num_bin)
        reshaped_back = reshaped_back.transpose(1, 0, 2)
        reshaped_back = reshaped_back[:, np.newaxis, :, :]
        return reshaped_back

    @staticmethod
    def zscore_blockwise(X: np.ndarray, block_wise: bool = False) -> np.ndarray:
        """Apply z-score normalization either blockwise or globally.
        
        Args:
            X: Input data with shape (batch, channel, height, width)
            block_wise: Whether to normalize in blocks of 8 electrodes
            
        Returns:
            Normalized data with same shape as input
        """
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input array, got shape {X.shape}")
            
        if block_wise:
            _, C, H, W = X.shape
            if H % 8 != 0:
                raise ValueError(f"Number of electrodes ({H}) must be divisible by 8")
                
            G = H // 8
            X_out = np.zeros_like(X, dtype=np.float32)
            for g in range(G):
                sl = slice(g * 8, (g + 1) * 8)
                block = X[:, :, sl, :]  
                μ = block.mean(axis=0, keepdims=True)
                σ = block.std(axis=0, keepdims=True, ddof=1)
                X_out[:, :, sl, :] = (block - μ) / (σ + 1e-8)
        else:
            μ = X.mean(axis=0, keepdims=True)
            σ = X.std(axis=0, keepdims=True, ddof=1)
            X_out = (X - μ) / (σ + 1e-8)
        return X_out

    @staticmethod
    def zscore_MAD(X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Apply robust z-score normalization using Median Absolute Deviation.
        
        Args:
            X: Input data with shape (batch, channel, height, width)
            eps: Small constant to prevent division by zero
            
        Returns:
            Normalized data with same shape as input
        """
        _, C, H, W = X.shape
        if H % 8 != 0:
            raise ValueError("electrode num must be divisible by 8")
            
        G = H // 8
        X_out = np.zeros_like(X, dtype=np.float32)
        for g in range(G):
            sl = slice(g * 8, (g + 1) * 8)
            block = X[:, :, sl, :]  
            mu = np.median(block, axis=0, keepdims=True)
            mad = np.median(np.abs(block - mu), axis=0, keepdims=True)
            sigma = 1.4826 * mad
            sigma[sigma < eps] = 1.0
            X_out[:, :, sl, :] = (block - mu) / sigma
        return X_out

    @staticmethod
    def normalize_spike_tensor(X: np.ndarray, block_wise: bool = False, pos_means: Optional[np.ndarray] = None, pos_stds: Optional[np.ndarray] = None,
                             neg_means: Optional[np.ndarray] = None, neg_stds: Optional[np.ndarray] = None, eps: float = 0) -> np.ndarray:
        """Normalize spike tensor data using z-score on non-zero values and add a baseline.
        
        Args:
            X: Input data with shape (batch, planes, wires=80, time_bins=50)
                Plane 0: Positive spike amplitudes
                Plane 1: Negative spike amplitudes
            block_wise: Whether to normalize in blocks of 8 wires (bundle-wise) or per channel
            pos_means: Pre-computed mean values for positive spikes
            pos_stds: Pre-computed std values for positive spikes
            neg_means: Pre-computed mean values for negative spikes
            neg_stds: Pre-computed std values for negative spikes
            eps: Small constant to prevent division by zero
            baseline: Constant value to add after normalization to set the mean
            
        Returns:
            Normalized data with same shape as input, preserving zero entries.
            X_out[:, 0] contains only positive values, X_out[:, 1] contains only negative values.
            The normalized data will have mean=baseline and std=1.
        """
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input array, got shape {X.shape}")
            
        B, P, W, T = X.shape
        if W % 8 != 0:
            raise ValueError(f"Number of wires ({W}) must be divisible by 8")
            
        X = X[:, 0:2]    
        X_out = np.zeros_like(X, dtype=np.float32)
        
        baseline = 0
        use_logp = False
        if block_wise:
            # Bundle-wise z-score normalization
            n_blocks = W // 8    
            for b in range(n_blocks):
                sl = slice(b * 8, (b + 1) * 8)
                block = X[:, :, sl, :]
                
                # Handle positive spikes (plane 0)
                pos_mask = block[:, 0] > 0
                if np.any(pos_mask):
                    pos_nonzero = block[:, 0][pos_mask]
                    if use_logp:
                        pos_nonzero = np.log1p(pos_nonzero)
                    if pos_means is not None:
                        pos_mean = pos_means[b]
                    else:
                        pos_mean = np.mean(pos_nonzero)
                    if pos_stds is not None:
                        pos_std = pos_stds[b]
                    else:
                        pos_std = np.std(pos_nonzero) + eps
                    # X_out[:, 0, sl, :][pos_mask] = pos_nonzero / pos_std
                    X_out[:, 0, sl, :][pos_mask] = (pos_nonzero - pos_mean) / pos_std + baseline
                
                # Handle negative spikes (plane 1)
                neg_mask = block[:, 1] < 0
                if np.any(neg_mask):
                    neg_nonzero = block[:, 1][neg_mask]
                    if use_logp:
                        neg_nonzero = -np.log1p(np.abs(neg_nonzero))
                    if neg_means is not None:
                        neg_mean = neg_means[b]
                    else:
                        neg_mean = np.mean(neg_nonzero)
                    if neg_stds is not None:
                        neg_std = neg_stds[b]
                    else:
                        neg_std = np.std(neg_nonzero) + eps
                    # X_out[:, 1, sl, :][neg_mask] = neg_nonzero / neg_std
                    X_out[:, 1, sl, :][neg_mask] = (neg_nonzero - neg_mean) / neg_std - baseline
        else:
            # Channel-wise z-score normalization
            for w in range(W):
                # Handle positive spikes (plane 0)
                pos_mask = X[:, 0, w, :] > 0
                if np.any(pos_mask):
                    pos_nonzero = X[:, 0, w, :][pos_mask]
                    if use_logp:
                        pos_nonzero = np.log1p(pos_nonzero)
                    if pos_means is not None:
                        pos_mean = pos_means[w]
                    else:
                        pos_mean = np.mean(pos_nonzero)
                    if pos_stds is not None:
                        pos_std = pos_stds[w]
                    else:
                        pos_std = np.std(pos_nonzero) + eps
                    # X_out[:, 0, w, :][pos_mask] = pos_nonzero / pos_std
                    X_out[:, 0, w, :][pos_mask] = (pos_nonzero - pos_mean) / pos_std + baseline
                
                # Handle negative spikes (plane 1)
                neg_mask = X[:, 1, w, :] < 0
                if np.any(neg_mask):
                    neg_nonzero = X[:, 1, w, :][neg_mask]
                    if use_logp:
                        neg_nonzero = -np.log1p(np.abs(neg_nonzero))
                    if neg_means is not None:
                        neg_mean = neg_means[w]
                    else:
                        neg_mean = np.mean(neg_nonzero)
                    if neg_stds is not None:
                        neg_std = neg_stds[w]
                    else:
                        neg_std = np.std(neg_nonzero) + eps
                    # X_out[:, 1, w, :][neg_mask] = neg_nonzero / neg_std
                    X_out[:, 1, w, :][neg_mask] = (neg_nonzero - neg_mean) / neg_std - baseline
        
        return X_out

    @staticmethod
    def channel_stats(data: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate channel-wise statistics for positive and negative planes after signed log1p transformation.
        
        Args:
            data: Input data with shape (batch, planes=2, wires=80, time_bins)
            eps: Small constant to prevent division by zero
            
        Returns:
            Tuple of (pos_means, pos_stds, neg_means, neg_stds) where each is an array of 
            statistics per channel
        """
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input array, got shape {data.shape}")
            
        b, p, w, t = data.shape
        pos_means = np.zeros(w)
        pos_stds = np.zeros(w)
        neg_means = np.zeros(w)
        neg_stds = np.zeros(w)
        
        # Calculate statistics for each channel
        for i in range(w):
            # Positive spikes (plane 0)
            pos_mask = data[:, 0, i, :] > 0
            if np.any(pos_mask):
                pos_nonzero = data[:, 0, i, :][pos_mask]
                # Apply signed log1p transformation
                # pos_nonzero = np.log1p(pos_nonzero)
                pos_means[i] = np.mean(pos_nonzero)
                pos_stds[i] = np.std(pos_nonzero) + eps
            
            # Negative spikes (plane 1)
            neg_mask = data[:, 1, i, :] < 0
            if np.any(neg_mask):
                neg_nonzero = data[:, 1, i, :][neg_mask]
                # Apply signed log1p transformation
                # neg_nonzero = -np.log1p(np.abs(neg_nonzero))  # Keep negative sign
                neg_means[i] = np.mean(neg_nonzero)
                neg_stds[i] = np.std(neg_nonzero) + eps
            
        return pos_means, pos_stds, neg_means, neg_stds

    @staticmethod
    def channel_max(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bundle-wise maximum values for positive and negative planes.
        
        Args:
            data: Input data with shape (batch, planes=2, wires=80, time_bins)
            
        Returns:
            Tuple of (pos_maxes, neg_maxes) where each is an array of maximum values per bundle
        """
        data = np.abs(data)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input array, got shape {data.shape}")
            
        b, p, w, t = data.shape
        if w % 8 != 0:
            raise ValueError(f"Number of wires ({w}) must be divisible by 8")
        
        n_blocks = w // 8
        pos_maxes = np.zeros(n_blocks)
        neg_maxes = np.zeros(n_blocks)
        
        # Calculate max values for each bundle
        for i in range(n_blocks):
            sl = slice(i * 8, (i + 1) * 8)
            block = data[:, :, sl, :]
            pos_maxes[i] = np.max(np.abs(block[:, 0]))  # Max for positive spikes
            neg_maxes[i] = np.max(np.abs(block[:, 1]))  # Max for negative spikes
            
        return pos_maxes, neg_maxes

    @staticmethod
    def signed_log1p(x: np.ndarray) -> np.ndarray:
        """Apply signed log1p transformation to the data.
        
        Args:
            x: Input data array
            
        Returns:
            Transformed data with same shape as input
        """
        return np.sign(x) * np.log1p(np.abs(x))

    def _normalize_spike_data(self, spike_data: np.ndarray, pos_means: Optional[np.ndarray] = None, pos_stds: Optional[np.ndarray] = None,
                            neg_means: Optional[np.ndarray] = None, neg_stds: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize spike data using different normalization methods.
        
        Args:
            spike_data: Raw spike data array with shape (batch, planes=4, wires=80, time_bins=50)
            pos_means: Pre-computed mean values for positive spikes
            pos_stds: Pre-computed std values for positive spikes
            neg_means: Pre-computed mean values for negative spikes
            neg_stds: Pre-computed std values for negative spikes
            
        Returns:
            Normalized spike data
        """
        norm_method = self.config.get('norm_method', 'zscore_channel')
        
        if norm_method not in ['zscore_channel', 'zscore_bundle']:
            raise ValueError(f"Unsupported normalization method: {norm_method}")

        return self.normalize_spike_tensor(spike_data, block_wise=(norm_method == 'zscore_bundle'),
                                            pos_means=pos_means, pos_stds=pos_stds, neg_means=neg_means, neg_stds=neg_stds)

    def load_clustless(self, files: List[str], pos_means: Optional[np.ndarray] = None, pos_stds: Optional[np.ndarray] = None,
                      neg_means: Optional[np.ndarray] = None, neg_stds: Optional[np.ndarray] = None) -> np.ndarray:
        """Load and preprocess spike data from multiple files.
        
        Args:
            files: List of paths to spike data files
            pos_means: Optional pre-computed mean values for positive spikes
            pos_stds: Optional pre-computed std values for positive spikes
            neg_means: Optional pre-computed mean values for negative spikes
            neg_stds: Optional pre-computed std values for negative spikes
            
        Returns:
            Preprocessed spike data
            
        Raises:
            ValueError: If no files provided
        """
        if not files:
            raise ValueError("No spike files provided")
            
        spike = []
        for file in files:
            data = np.load(file)['data']
            spike.append(data[:, :, None])
        spike = np.concatenate(spike, axis=2)

        return self._normalize_spike_data(spike, pos_means, pos_stds, neg_means, neg_stds)
        # return spike

    def load_lfp(self, files: List[str]) -> np.ndarray:
        """Load and preprocess LFP data from multiple files.
        
        Args:
            files: List of paths to LFP data files
            
        Returns:
            Preprocessed LFP data
            
        Raises:
            ValueError: If no files provided
        """
        if not files:
            raise ValueError("No LFP files provided")
            
        lfp = []
        for file in files:
            data = np.load(file)['data']
            region = file.split('marco_lfp_spectrum_')[-1].split('.npz')[0]
            if region == "WM":
                continue
            self.lfp_channel_by_region[region] = data.shape[1]
            
            if len(data.shape) == 2:
                lfp.append(data[:, None, :])
            else:
                lfp.append(data)

        lfp = np.concatenate(lfp, axis=1)
        return lfp[:, None]

    def preprocess_data(self) -> None:
        """Perform additional data preprocessing steps.
        
        This method should be overridden by child classes to implement
        specific preprocessing steps.
        """
        pass

    def _filter_files_by_regions(self, files: List[str], regions: List[str], config: Dict[str, Any]) -> List[str]:
        """Filter files to only include those matching specified regions.
        
        Args:
            files: List of file paths
            regions: List of region names to include
            config: Configuration dictionary containing:
                - lesion: Optional lesion type
                - lesion_mode: Optional filtering mode ('only' or 'without')
            
        Returns:
            Filtered list of file paths
            
        Example:
            >>> files = ['path/to/MTL1.npz', 'path/to/FC1.npz']
            >>> regions = ['MTL']
            >>> config = {'lesion': 'MTL', 'lesion_mode': 'only'}
            >>> filtered = self._filter_files_by_regions(files, regions, config)
            # Returns only MTL files
        """
        # Filter files based on regions
        filtered_files = [item for item in files if any(sub in item for sub in regions)]
        return filtered_files

    def _get_good_sample_indices(self, occurrence_threshold: int = 200) -> List[int]:
        """Get indices of samples that meet occurrence threshold.
        
        Args:
            occurrence_threshold: Minimum number of occurrences required
            
        Returns:
            Sorted list of good sample indices
            
        Raises:
            ValueError: If self.label is not set
        """
        if not hasattr(self, 'label') or len(self.label) == 0:
            raise ValueError("Labels must be loaded before calling _get_good_sample_indices")

        # For movie dataset, we look at first 8 columns only
        if hasattr(self, 'ml_label'):  # This is a movie dataset
            label_data = self.label[0][:, 0:8]
        else:  # This is a free recall dataset
            label_data = self.label[0]

        class_value, class_count = np.unique(label_data, axis=0, return_counts=True)
        good_indices = np.where(class_count >= occurrence_threshold)[0]
        
        indices = []
        for index in good_indices:
            label = class_value[index]
            label_indices = np.where((label_data == label[None, :]).all(axis=1))[0]
            indices.extend(label_indices)
            
        return sorted(indices)

class MyDataset(Dataset):
    """PyTorch Dataset for handling neural data.
    
    This dataset can handle either LFP data, spike data, or both combined.
    It supports data transformations and maintains sample weights for imbalanced data.
    """
    
    def __init__(
        self, 
        lfp_data: Optional[np.ndarray] = None,
        spike_data: Optional[np.ndarray] = None,
        label: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        pos_weight: Optional[np.ndarray] = None
    ) -> None:
        """Initialize the dataset.
        
        Args:
            lfp_data: LFP data array if available
            spike_data: Spike data array if available
            label: Labels for each sample
            indices: Original indices of samples for tracking
            transform: Optional transform to be applied to the data
            pos_weight: Class weights for handling imbalanced data
            
        Raises:
            ValueError: If neither LFP nor spike data is provided
            ValueError: If label is None
            ValueError: If indices is None
        """
        if label is None:
            raise ValueError("Labels must be provided")
        if indices is None:
            raise ValueError("Sample indices must be provided")
        if lfp_data is None and spike_data is None:
            raise ValueError("At least one of lfp_data or spike_data must be provided")

        self.lfp_data = lfp_data
        self.spike_data = spike_data
        self.label = label
        self.transform = transform
        self.pos_weight = pos_weight
        self.indices = indices

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.label)

    def __getitem__(self, index: int) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to get
            
        Returns:
            A tuple containing:
                - The neural data (either LFP, spike, or both)
                - The label for this sample
                - The original index of this sample
        """
        label = self.label[index]
        idx = self.indices[index]

        if self.lfp_data is not None and self.spike_data is None:
            data = self.lfp_data[index]
        elif self.lfp_data is None and self.spike_data is not None:
            data = self.spike_data[index]
        else:
            data = (self.lfp_data[index], self.spike_data[index])

        if self.transform is not None:
            data = self.transform(data)

        return data, label, idx

def create_loader(
    dataset: Union[MyDataset, BaseNeuralDataset],
    batch_size: int,
    sampler: Optional[Sampler] = None,
    shuffle: bool = False,
    num_workers: int = 1,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader with specified parameters.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        sampler: Optional sampler for the data
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in GPU applications
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle and sampler is None
    ) 