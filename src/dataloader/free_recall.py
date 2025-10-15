import numpy as np
import pandas as pd
import os
import pickle
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch
import re
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms
from .neural_data import BaseNeuralDataset, MyDataset
from typing import Optional, List, Dict, Any
import random
from src.param.param_data import *


class FreeRecallDataset(BaseNeuralDataset):
    """Dataset class for handling free recall experiment data.
    
    This class processes neural recordings during free recall experiments.
    It inherits common functionality from BaseNeuralDataset and adds
    free recall-specific features. This is an inference-only dataset.
    """
    
    REQUIRED_CONFIG_KEYS = {
        'patient', 'use_spike', 'use_lfp', 'use_overlap',
        'use_combined', 'free_recall_phase'
    }
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the free recall dataset.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If required config keys are missing
        """
        # Validate config
        missing_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        if config['use_spike'] and 'spike_path' not in config:
            raise ValueError("spike_path must be provided when use_spike is True")
            
        if config['use_lfp'] and 'lfp_path' not in config:
            raise ValueError("lfp_path must be provided when use_lfp is True")

        # Initialize base class
        super().__init__(config)
        
        # Free recall-specific initialization
        self.phase = config['free_recall_phase']
        self.use_sleep = config.get('use_sleep', False)
        self.lfp_data_mode = config.get('lfp_data_mode', '')
        self.data_version = config.get('data_version', '')
        
        # Load data
        self._load_data()
        
        print("Free Recall Data Loaded")
        self.preprocess_data()
        
        # Apply data augmentation if configured
        if config.get('use_shuffle_diagnostic'):
            self.circular_shift()

    def _load_data(self) -> None:
        """Load spike and LFP data based on configuration."""
        # Load spike data
        if self.use_spike:
            if self.use_sleep:
                self._load_sleep_spike_data()
            else:
                self._load_recall_spike_data()
            self.data.append(self.spike_data)

        # Load LFP data
        if self.use_lfp:
            if self.use_sleep:
                self._load_sleep_lfp_data()
            else:
                self._load_recall_lfp_data()
            self.data.append(self.lfp_data)
        
        # Combine data if needed
        if self.use_combined:
            self.data = {'clusterless': self.data[0], 'lfp': self.data[1]}
        else:
            self.data = self.data[0]
            
        # Clean up
        del self.lfp_data
        del self.spike_data

    def _load_sleep_spike_data(self) -> None:
        """Load spike data for sleep phase."""
        version = self.data_version
        regions_list = list(dict.fromkeys(SPIKE_REGION[self.patient]))


        spike_path = os.path.join(self.config['spike_path'], self.patient, version, 'time_sleep_1')

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
        self.spike_data = self.load_clustless(spike_files)


    def _load_recall_spike_data(self) -> None:
        """Load spike data for recall phase."""
        version = self.data_version
        regions_list = list(dict.fromkeys(SPIKE_REGION[self.patient]))

        if 'all' in self.phase:
            phases = ['FR1a', 'FR1b'] if self.patient == 'i728' else ['FR1', 'CR1']
            spike_data_list = []
            for phase in phases:
                spike_path = os.path.join(self.config['spike_path'], self.patient, version, f'time_recall_{phase}')
                spike_files = self._get_sorted_files(spike_path)
                regions = self._get_filtered_regions(self.config)
                spike_files = self._filter_files_by_regions(spike_files, regions, self.config)
                spike_data = self.load_clustless(spike_files)
                spike_data_list.append(spike_data)
            self.spike_data = np.concatenate(spike_data_list, axis=0)
            
        elif 'control' in self.phase:
            spike_path = os.path.join(self.config['spike_path'], self.patient, 
                                    version, f'time_{self.phase}')
            spike_files = self._get_sorted_files(spike_path)
            regions = self._get_filtered_regions(self.config)
            spike_files = self._filter_files_by_regions(spike_files, regions, self.config)
            self.spike_data = self.load_clustless(spike_files)
            
        elif 'movie' in self.phase:
            spike_path = os.path.join(self.config['spike_path'], self.patient, 
                                    version, 'time_movie_1')
            spike_files = self._get_sorted_files(spike_path)
            regions = self._get_filtered_regions(self.config)
            spike_files = self._filter_files_by_regions(spike_files, regions, self.config)
            self.spike_data = self.load_clustless(spike_files)
        else:
            spike_path = os.path.join(self.config['spike_path'], self.patient, 
                                    version, f'time_recall_{self.phase}')

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
            self.spike_data = self.load_clustless(spike_files)

    def _load_sleep_lfp_data(self) -> None:
        """Load LFP data for sleep phase."""
        lfp_path = os.path.join(self.config['lfp_path'], self.patient, 
                               '', 'spectrogram_sleep')
        lfp_files = self._get_sorted_files(lfp_path)
        self.lfp_data = self.load_lfp(lfp_files)

    def _load_recall_lfp_data(self) -> None:
        """Load LFP data for recall phase."""
        version = self.lfp_data_mode
        
        if 'all' in self.phase:
            phases = [1, 3] if self.patient == 'i728' else [1, 2]
            lfp_data_list = []
            for phase in phases:
                lfp_path = os.path.join(self.config['lfp_path'], self.patient, 
                                      version, f'spectrogram_recall_{phase}')
                lfp_files = self._get_sorted_files(lfp_path)
                lfp_data = self.load_lfp(lfp_files)
                lfp_data_list.append(lfp_data)
            self.lfp_data = np.concatenate(lfp_data_list, axis=0)
            
        elif 'control' in self.phase:
            lfp_path = os.path.join(self.config['lfp_path'], self.patient, 
                                  version, f'spectrogram_{self.phase}')
            lfp_files = self._get_sorted_files(lfp_path)
            self.lfp_data = self.load_lfp(lfp_files)
            
        else:
            lfp_path = os.path.join(self.config['lfp_path'], self.patient, 
                                  version, f'spectrogram_recall_{self.phase}')
            lfp_files = self._get_sorted_files(lfp_path)
            
            if self.config.get('lesion') == 'MTL':
                regions = ['HPC', 'AMY', 'PHC', 'ERC']
                regions = [f"{item}.npz" for item in regions]
                lfp_files = self._filter_files_by_regions(lfp_files, regions, self.config)
            
            self.lfp_data = self.load_lfp(lfp_files)

    def preprocess_data(self) -> None:
        """Calculate and store the dataset length."""
        if self.use_combined:
            assert self.data['clusterless'].shape[0] == self.data['lfp'].shape[0]
            self.data_length = self.data['clusterless'].shape[0]
        else:
            self.data_length = self.data.shape[0]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data_length

    def circular_shift(self) -> None:
        """Apply circular shift augmentation to the data."""
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Circular shift only supports single data type (not combined)")
            
        if self.data.ndim != 4:
            raise ValueError(f"Expected 4D data array, got shape {self.data.shape}")
            
        if not isinstance(self.gap, (int, float)) or self.gap < 0:
            raise ValueError(f"Invalid gap parameter: {self.gap}")
            
        b, c, h, w = self.data.shape
        shift_amount = self.gap
        print(self.patient, shift_amount)
        
        for i in range(self.data.shape[2]):
            local_random = random.Random(shift_amount + i)
            shift_amount = local_random.randint(120, 7600)
            self.data[:, :, i, :] = np.roll(self.data[:, :, i, :], shift=shift_amount, axis=0)

    def _process_labels(self, indices: List[int]) -> None:
        """Process and filter labels based on indices."""
        self.label = np.concatenate(self.label, axis=0)[indices]

    def visualization(self):
        combined_bins = np.vstack((self.data, self.labels))
        combined_bins = self.normalize_bins(combined_bins)
        figpath = "./bins.png"

        plt.figure()
        plt.imshow(combined_bins, aspect='auto', interpolation='nearest')
        plt.savefig(figpath)
        plt.show()


class MyDataset(Dataset):
    """Dataset class for handling LFP and spike data.
    
    This class manages the joint or individual loading of LFP and spike data
    for neural processing tasks.
    
    Args:
        lfp_data: Optional array of LFP (Local Field Potential) data
        spike_data: Optional array of spike data
        label: Optional array of labels (unused in current implementation)
        indices: List of indices for data access
        transform: Optional transform to be applied to the data
    """
    
    def __init__(self, 
                 lfp_data: Optional[np.ndarray] = None,
                 spike_data: Optional[np.ndarray] = None,
                 label: Optional[np.ndarray] = None,
                 indices: List[int] = None,
                 transform: Optional[callable] = None):
        self.lfp_data = lfp_data
        self.spike_data = spike_data
        self.transform = transform
        self.indices = indices or []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            A tuple of (data, index) where data can be:
            - (lfp_data, spike_data) if both are available
            - lfp_data if only LFP is available
            - spike_data if only spike data is available
        """
        idx = self.indices[index]
        
        # Handle single data type cases
        if self.lfp_data is not None and self.spike_data is None:
            data = self.lfp_data[idx]
            if self.transform:
                data = self.transform(data)
            return data, idx
            
        if self.lfp_data is None and self.spike_data is not None:
            data = self.spike_data[idx]
            if self.transform:
                data = self.transform(data)
            return data, idx
        
        # Handle combined data case
        lfp = self.lfp_data[idx]
        spike = self.spike_data[idx]
        
        if self.transform:
            lfp = self.transform(lfp)
            spike = self.transform(spike)
            
        return (lfp, spike), idx


def create_inference_combined_loaders(
        dataset: FreeRecallDataset,
        config: Dict[str, Any],
        batch_size: int = 128,
        seed: Optional[int] = 42,
        batch_sample_num: int = 2048,
        shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for inference.
    
    Args:
        dataset: The dataset to create a loader from
        config: Configuration dictionary
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        batch_sample_num: Number of samples per epoch
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader configured for inference
    """
    # if seed is not None:
    #     np.random.seed(seed)

    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    if shuffle:
        # np.random.seed(seed)
        np.random.shuffle(all_indices)

    if config['use_combined']:
        spike_inference = dataset.data['clusterless'][all_indices]
        lfp_inference = dataset.data['lfp'][all_indices]
    else:
        spike_inference = dataset.data[all_indices] if config['use_spike'] else None
        lfp_inference = dataset.data[all_indices] if config['use_lfp'] else None

    inference_dataset = MyDataset(lfp_inference, spike_inference, None, all_indices)

    return DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=False,
        shuffle=shuffle,
    )
