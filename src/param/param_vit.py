import numpy as np 
import os


param_dict={
    'lr': 1e-04,
    'batch_size': 128,
    'weight_decay': 1e-04,
    'lr_drop': 50,
    'num_labels': 8,
    'num_channels': 2,
    'merge_label': True,
    'patch_size': (1, 5),
    # path
    'movie_label_path': 'data/8concepts_movie_label.npy',
    'spike_path': 'data/spike_data',
}