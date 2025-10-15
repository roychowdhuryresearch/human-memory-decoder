import numpy as np
import pandas as pd
import copy
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
from src.param.param_data import *
import random
from typing import Optional, Callable, Tuple, Union, Dict, Any, List
from .neural_data import BaseNeuralDataset, MyDataset

class MovieDataset(BaseNeuralDataset):
    """Dataset class for handling movie experiment data.
    
    This class processes neural recordings during movie viewing experiments.
    It inherits common functionality from BaseNeuralDataset and adds
    movie-specific features.
    """
    
    REQUIRED_CONFIG_KEYS = {
        'patient', 'use_spike', 'use_lfp', 'use_overlap',
        'use_combined', 'data_version', 'gap', 'label_path'
    }
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the movie dataset.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If required config keys are missing
            FileNotFoundError: If required paths don't exist
        """
        # Validate config
        missing_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Validate paths exist
        if not os.path.exists(config['label_path']):
            raise FileNotFoundError(f"Label path not found: {config['label_path']}")
        
        if config['use_spike'] and 'spike_path' not in config:
            raise ValueError("spike_path must be provided when use_spike is True")
            
        if config['use_lfp'] and 'lfp_path' not in config:
            raise ValueError("lfp_path must be provided when use_lfp is True")

        # Initialize base class
        super().__init__(config)
        
        # Movie-specific initialization
        self.data_version = config['data_version']
        self.gap = config['gap']
        self.movie_sampling_rate = 30  # Movie sampling rate in Hz
        self.resolution = 4  # Temporal resolution factor
        self.label_path = config['label_path']
        
        # Load and process labels
        self.labels = self._load_movie_labels(self.label_path)
        # Repeat labels to match temporal resolution
        self.labels = np.repeat(self.labels, self.resolution, axis=1)
        
        # Initialize data containers
        self.label = []
        self.smoothed_label = []
        
        # Load data
        self._load_data()
                
        print("Movie Data Loaded")
        self.preprocess_data()
        
        # Apply data augmentation if configured
        if config.get('use_shuffle_diagnostic'):
            self.circular_shift()

    def _load_movie_labels(self, path: str) -> np.ndarray:
        """Load and preprocess movie labels from file."""
        try:
            labels = np.load(path)
            if labels.size == 0:
                raise ValueError("Empty movie label file")
            return labels
        except (IOError, ValueError) as e:
            raise IOError(f"Failed to load movie labels from {path}: {e}")

    def _load_data(self) -> None:
        """Load spike and LFP data based on configuration."""
        categories = self._get_data_categories()
        regions_list = list(dict.fromkeys(SPIKE_REGION[self.patient]))
        # Load spike data
        if self.use_spike:
            for category in categories:
                version = self.data_version
                spike_path = os.path.join(self.config['spike_path'], self.patient, 
                                        version, f'time_{category.lower()}')
                spike_files_all = self._get_sorted_files(spike_path)
                
                spike_files = []
                for region in regions_list:
                    sub_regions = self.get_localization_clusterless_by_region(self.patient, region)

                    if self.config.get('lesion') != 'Full':
                        lesion_regions = self.get_localization_clusterless(self.patient, self.config['lesion'])
                        mode = 'include' if self.config.get('lesion_mode') == 'only' else 'exclude'
                        if mode == 'include' and set(sub_regions).isdisjoint(set(lesion_regions)):
                            continue
                        elif mode == 'exclude' and set(sub_regions).issubset(set(lesion_regions)):
                            continue

                    sub_regions_files = [f"{item}{i}" for item in sub_regions for i in range(1, 9)]
                    file_names = [item for item in spike_files_all if any(sub in item for sub in sub_regions_files)]
                    spike_files.extend(file_names)
                    self.spike_channel_by_region[region] = len(file_names)
                
                print(f'Model: was {len(spike_files_all)} channels, now {len(spike_files)} channels')
                
                spike_data = self.load_clustless(spike_files)
                self.spike_data.append(spike_data)
                
            self.spike_data = np.concatenate(self.spike_data, axis=0)
            self.data.append(self.spike_data)

        # Load LFP data
        if self.use_lfp:
            for category in categories:
                version = self.config.get('lfp_data_mode', '')
                lfp_path = os.path.join(self.config['lfp_path'], self.patient, 
                                      version, f'spectrogram_{category.lower()}')
                
                lfp_files = self._get_sorted_files(lfp_path)
                if self.config.get('lesion') == 'MTL':
                    regions = ['HPC', 'AMY', 'PHC', 'ERC']
                    regions = [f"{item}.npz" for item in regions]
                    lfp_files = self._filter_files_by_regions(lfp_files, regions, self.config)
                
                lfp_data = self.load_lfp(lfp_files)
                self.lfp_data.append(lfp_data)
                
            self.lfp_data = np.concatenate(self.lfp_data, axis=0)
            self.data.append(self.lfp_data)
        
        # Process labels
        self.label.append(self.labels.transpose().astype(np.float32))
        self.smoothed_label.append(self.labels.transpose().astype(np.float32))
        
        # Process data
        good_indices = self._get_good_sample_indices()
        self._process_labels(good_indices)
        self._process_data(good_indices)

    def _get_data_categories(self) -> List[str]:
        """Get list of data categories based on patient ID."""
        if self.patient in ['564', '565']:
            return ['Movie_1', 'Movie_2']
        return ['Movie_1']

    def _process_labels(self, indices: List[int]) -> None:
        """Process and filter labels based on indices."""
        self.label = np.concatenate(self.label, axis=0)[indices]
        self.smoothed_label = np.concatenate(self.smoothed_label, axis=0)[indices]

    def _process_data(self, indices: List[int]) -> None:
        """Process and filter data based on indices.
        
        Args:
            indices: List of indices to keep
        """
        if self.use_combined:
            self.data = {
                'clusterless': self.data[0][indices],
                'lfp': self.data[1][indices]
            }
        else:
            self.data = self.data[0][indices]
        
        # Clean up
        del self.lfp_data
        del self.spike_data

    def smooth_label(self) -> np.ndarray:
        """Apply Gaussian smoothing to the movie labels."""
        sigma = 1
        kernel = np.exp(-(np.arange(-1, 2) ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        
        smoothed_label = convolve1d(self.labels, kernel, axis=1)
        return np.round(smoothed_label, 2)

    def time_backword(self) -> None:
        """Augment data by flipping time dimension."""
        flipped = np.flip(self.data, axis=-1)
        self.data = np.concatenate((self.data, flipped), axis=0)
        self.label = np.repeat(self.label, 2, axis=0)
        self.smoothed_label = np.repeat(self.smoothed_label, 2, axis=0)

    def circular_shift(self) -> None:
        """Apply circular shift augmentation to the data."""
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Circular shift only supports single data type (not combined)")
            
        if self.data.ndim != 4:
            raise ValueError(f"Expected 4D data array, got shape {self.data.shape}")
            
        if not isinstance(self.gap, (int, float, np.integer)) or self.gap < 0:
            raise ValueError(f"Invalid gap parameter: {self.gap}")
            
        b, c, h, w = self.data.shape
        shift_amount = self.gap
        print(self.patient, shift_amount)
        
        for i in range(self.data.shape[2]):
            # local_random = random.Random(shift_amount + i)
            # shift_amount = local_random.randint(120, 7600)
            self.data[:, :, i, :] = np.roll(self.data[:, :, i, :], shift=shift_amount, axis=0)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.label)

def split_by_class_value(dataset, p_val, seed=42):
    labels = dataset.label[:, 0:8]
    class_values, class_indices = np.unique(labels, axis=0, return_inverse=True)
    train_indices = []
    val_indices = []
    for class_id in range(len(class_values)):
        idx = np.where(class_indices == class_id)[0]
        n_val = int(np.ceil(len(idx) * p_val))
        if n_val == 0 and len(idx) > 0:
            n_val = 1  # Ensure at least one sample in val if possible
        if len(idx) > 0:
            val_idx = np.random.choice(idx, size=n_val, replace=False)
            train_idx = np.setdiff1d(idx, val_idx, assume_unique=True)
            val_indices.extend(val_idx)
            train_indices.extend(train_idx)
    return np.array(train_indices), np.array(val_indices)

def create_weighted_loaders(
    dataset: MovieDataset,
    config: Dict[str, Any],
    batch_size: int = 128,
    seed: int = 42,
    p_val: float = 0.1,
    batch_sample_num: int = 2048,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    extras: Dict[str, Any] = {},
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders with weighted sampling for training, validation, and testing.

    Args:
        dataset: The neural dataset to create loaders from
        config: Configuration dictionary
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        p_val: Proportion of data to use for validation (0 to 1)
        batch_sample_num: Number of samples per epoch
        shuffle: Whether to shuffle the data
        transform: Optional transform to apply to the data
        extras: Additional configuration options

    Returns:
        A tuple of (train_loader, val_loader, test_loader)
        val_loader and test_loader may be None depending on p_val

    Raises:
        ValueError: If p_val is not between 0 and 1
        AssertionError: If data splitting produces invalid results
    """
    if p_val < 0 or p_val >= 1:
        raise ValueError("p_val must be between 0 and 1")

    # # Set random seed for reproducibility
    # if seed is not None:
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)

    dataset_size = len(dataset)
    all_indices = np.arange(dataset_size)

    # Calculate class weights
    class_value, class_count = np.unique(dataset.label[:, 0:8], axis=0, return_counts=True)
    class_weight_dict = {key.tobytes(): dataset_size / value for key, value in zip(class_value, class_count)}
    data_weights = np.array([class_weight_dict[label.tobytes()] for label in dataset.label[:, 0:8]])

    # Split data into train/val if p_val > 0
    if p_val > 0:
        train_indices, val_indices = split_by_class_value(dataset, p_val, seed)
        
        # Verify split integrity
        assert len(set(val_indices)) + len(set(train_indices)) == len(all_indices), "Data split mismatch"
        
        # Save split information
        # _save_split_data(dataset, config, train_indices, val_indices)
        
        # Create datasets
        train_dataset, val_dataset = _create_train_val_datasets(
            dataset, train_indices, val_indices, transform, data_weights
        )
        
        # Create data loaders
        train_loader = _create_loader(
            train_dataset,
            batch_size,
            WeightedRandomSampler(
                weights=data_weights[train_indices],
                num_samples=batch_sample_num,
                replacement=True
            ),
            shuffle=False
        )
        
        val_loader = _create_loader(
            val_dataset,
            batch_size,
            sampler=None,
            shuffle=False
        )
        
        return train_loader, val_loader, None
    
    else:  # No validation split
        train_indices = all_indices

        if shuffle:
            np.random.shuffle(train_indices)

        train_dataset = MyDataset(
            dataset.data[train_indices] if config['use_lfp'] else None,
            dataset.data[train_indices] if config['use_spike'] else None,
            dataset.smoothed_label[train_indices],
            train_indices,
            transform=transform,
            pos_weight=_calculate_pos_weights(dataset.smoothed_label[train_indices])
        )
        
        train_loader = _create_loader(
            train_dataset,
            batch_size,
            WeightedRandomSampler(
                weights=data_weights[train_indices],
                num_samples=batch_sample_num,
                replacement=True
            ),
            shuffle=False
        )
        
        return train_loader, None, None

def _split_indices_by_chunks(dataset: MovieDataset, p_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into training and validation sets while preserving temporal chunks.
    
    Args:
        dataset: The dataset to split
        p_val: Proportion of data to use for validation
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    # Convert labels to strings for unique combination identification
    tag_combinations = np.apply_along_axis(lambda x: ''.join(map(str, x)), 1, dataset.label)
    unique_combinations, indices = np.unique(tag_combinations, return_inverse=True)
    grouped_indices = {tag: np.where(indices == i)[0] for i, tag in enumerate(unique_combinations)}
    
    train_indices = []
    val_indices = []
    
    for idx_group in grouped_indices.values():
        chunks = np.split(idx_group, np.where(np.diff(idx_group) != 1)[0] + 1)
        for chunk in chunks:
            if len(chunk) > 1:
                val_size = max(1, int(np.ceil(len(chunk) * p_val)))
                val_indices.extend(chunk[:val_size])
                val_indices.extend(chunk[-val_size:])
                if len(chunk) > 2:
                    train_indices.extend(chunk[val_size:-val_size])
            else:
                val_indices.extend(chunk)
                
    return np.array(train_indices), np.array(val_indices)

def _save_split_data(dataset: MovieDataset, config: Dict[str, Any], 
                    train_indices: np.ndarray, val_indices: np.ndarray) -> None:
    """Save split data to disk.
    
    Args:
        dataset: The dataset being split
        config: Configuration dictionary
        train_indices: Indices for training set
        val_indices: Indices for validation set
    """
    # Save labels
    np.save(os.path.join(config['test_save_path'], 'train_label'), dataset.label[train_indices])
    np.save(os.path.join(config['test_save_path'], 'val_label'), dataset.label[val_indices])
    
    # Save data based on configuration
    if config['use_lfp'] and not config['use_combined']:
        np.save(os.path.join(config['test_save_path'], 'val_lfp'), dataset.data[val_indices])
    elif config['use_spike'] and not config['use_combined']:
        np.save(os.path.join(config['test_save_path'], 'val_clusterless'), dataset.data[val_indices])
    elif config['use_combined']:
        np.save(os.path.join(config['test_save_path'], 'val_lfp'), dataset.data[val_indices])
        np.save(os.path.join(config['test_save_path'], 'val_clusterless'), dataset.data[val_indices])

def _create_train_val_datasets(dataset: MovieDataset, train_indices: np.ndarray, 
                             val_indices: np.ndarray, transform: Optional[Callable],
                             data_weights: np.ndarray) -> Tuple[MyDataset, MyDataset]:
    """Create training and validation datasets.
    
    Args:
        dataset: The source dataset
        train_indices: Indices for training set
        val_indices: Indices for validation set
        transform: Optional transform to apply
        data_weights: Sample weights
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Get data based on indices
    train_data = dataset.data[train_indices] if dataset.data is not None else None
    val_data = dataset.data[val_indices] if dataset.data is not None else None
    
    # Get labels
    train_label = dataset.smoothed_label[train_indices]
    val_label = dataset.smoothed_label[val_indices]
    
    # Calculate positive weights
    train_pos_weights = _calculate_pos_weights(train_label)
    val_pos_weights = _calculate_pos_weights(val_label)
    
    # Create datasets
    train_dataset = MyDataset(train_data, None, train_label, train_indices, 
                            transform=transform, pos_weight=train_pos_weights)
    val_dataset = MyDataset(val_data, None, val_label, val_indices, 
                          pos_weight=val_pos_weights)
    
    return train_dataset, val_dataset

def _calculate_pos_weights(labels: np.ndarray) -> np.ndarray:
    """Calculate positive class weights for handling class imbalance.
    
    Args:
        labels: Label array
        
    Returns:
        Array of positive class weights
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = labels.shape[0] - pos_counts
    return neg_counts / pos_counts

def _create_loader(dataset: MyDataset, batch_size: int, sampler: Optional[Sampler] = None,
                  shuffle: bool = False) -> DataLoader:
    """Create a DataLoader with specified parameters.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        sampler: Optional sampler for the data
        shuffle: Whether to shuffle the data
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        shuffle=shuffle and sampler is None
    )

def create_inference_loaders(
    dataset: MovieDataset,
    batch_size: int = 128,
    seed: Optional[int] = 42,
    shuffle: bool = False,
    num_workers: int = 1,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for inference.

    Args:
        dataset: The dataset to create a loader from
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility (optional)
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in GPU applications

    Returns:
        DataLoader configured for inference
    """
    # if seed is not None:
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
