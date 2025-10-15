import numpy as np
import torch
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.initializer import *
from itertools import product
from src.trainer import Trainer
from src.utils.visualization import method_heatmap, method_MCS, method_curve_shape, method_curve_shape_all
import csv
from collections import defaultdict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you’re using multi-GPU
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_configs():
    """Create a list of configurations to run."""
    # Define patient exclusions for specific lesion configurations
    patient_exclusions = {
        ('MTL', 'without'): ['p4'],  # Exclude p4 for MTL-without
        ('FC', 'only'): ['p4'],      # Exclude p4 for FC-only
        ('HPC', 'only'): ['p9'],      # Exclude p9 for HPC-only
    }

    base_config = {
        # 'seed': 42,
        'device': 'cuda:3',
        'model_aggregate_type': 'mean2',
        'use_augment': False,
        'use_long_input': False,
        'use_shuffle_diagnostic': False,
        'use_overlap': False,
        'use_combined': False,
        'use_sleep': False,
        'shuffle': False,
        'gap': False,
        'label_number': 8,
        'label_path': 'data/8concepts_movie_label.npy',
        # all patients: 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'
        'patient': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'], 
        'epochs': 50,
        'save_epochs': [49],

        # architecture
        'num_csa_layers': 6,
        'num_cca_layers': 6,
        'hidden_size': 384,
        'num_attention_heads': 6,

        # data
        'norm_method': 'zscore_bundle',
        'data_type': 'clusterless',
        'data_version': 'simulated',
    }

    # Define variations for different parameters
    variations = {
        'lesion': ['MTL', 'HPC', 'FC'],
        'lesion_mode': ['without'],
    }

    # Generate all combinations
    configs = []

    # Generate all combinations with other parameters
    for values in product(variations['lesion'], variations['lesion_mode']):
        config = base_config.copy()
        lesion, lesion_mode = values[0], values[1]
        
        # Skip invalid combinations early
        if lesion == 'Full' and lesion_mode != 'only':
            continue

        # Set non-architecture parameters
        config.update({
            'lesion': lesion,
            'lesion_mode': lesion_mode
        })
            
        # Handle patient exclusions
        if (lesion, lesion_mode) in patient_exclusions:
            config['patient'] = [p for p in base_config['patient'] 
                               if p not in patient_exclusions[(lesion, lesion_mode)]]

        configs.append(config)
    
    return configs

def pipeline(config):
    set_seed(42)

    # Set device
    device = torch.device(config['device'])

    # Initialize components
    dataloaders = initialize_dataloaders(config)
    model = initialize_model(config).to(device)

    # Parameter count
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters}")

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_drop'])

    # Evaluator and Trainer
    evaluator = initialize_evaluator(config, 1)
    trainer = Trainer(model, evaluator, optimizer, lr_scheduler, dataloaders, config)

    return trainer

def build_save_paths(root_path, version, patient, data_type, architecture, suffix):
    """Build save paths for model outputs."""
    base_path = Path(root_path) / 'results' / version / f"{patient}_{data_type}_{architecture}_{suffix}"
    paths = {
        'train': base_path / 'train',
        'valid': base_path / 'valid',
        'test': base_path / 'test',
        'memory': base_path / 'memory'
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def build_args(config, patient, use_clusterless, use_lfp, use_combined, model_architecture, save_paths):
    # Start with base architecture-specific defaults
    args = initialize_configs(architecture=model_architecture)

    # Merge config directly
    args.update(config)

    # Override or add fields derived per patient/task
    args.update({
        'patient': patient,
        'use_spike': use_clusterless,
        'use_lfp': use_lfp,
        'use_combined': use_combined,
        'model_architecture': model_architecture,
        'use_shuffle': use_clusterless,
        'train_save_path': str(save_paths['train']),
        'valid_save_path': str(save_paths['valid']),
        'test_save_path': str(save_paths['test']),
        'memory_save_path': str(save_paths['memory']),
    })

    return args

def get_model_flags(data_type):
    """Get model flags based on data type."""
    if data_type == 'clusterless':
        return True, False, False, 'transformer'
    elif data_type == 'lfp':
        return False, True, False, 'transformer'
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def perform_memory_test(config, phase='recall1', alongwith=[], save_predictions=False):
    """Run memory tests on trained model."""
    # Get model flags and paths first
    use_clusterless, use_lfp, use_combined, model_architecture = get_model_flags(config['data_type'])
    
    # Create suffix based on data version and lesion configuration
    lesion_suffix = f'{config["lesion_mode"]}_{config["lesion"]}' if config['lesion'] != 'Full' else 'Full'
    suffix = lesion_suffix
    
    # Build version string
    version = get_version_string(config['num_csa_layers'], config['num_cca_layers'], config['hidden_size'], config['num_attention_heads'])

    # Start with base architecture-specific defaults
    args = initialize_configs(architecture=model_architecture)
    args.update(config)
    
    args.update({
        'free_recall_phase': phase,
        'use_spontaneous': False,
        'use_bipolar': False,
        'use_spike': use_clusterless,
        'use_lfp': use_lfp,
        'use_shuffle': use_clusterless,
        'use_sleep': False,
        'use_shuffle_diagnostic': False,
        'model_architecture': model_architecture
    })

    if config['patient'] == 'p10' and '1' in phase:
        args['free_recall_phase'] = 'FR1a'
        dataloaders = initialize_inference_dataloaders(args)
    else:
        dataloaders = initialize_inference_dataloaders(args)

    model = initialize_model(args)
    model = model.to(args['device'])
    
    # Build model path using the same structure as in training
    root_path = './'
    if config['epoch'] == 'best':
        model_dir = os.path.join(root_path, 'results', version, 
                                f"{config['patient']}_{config['data_type']}_{model_architecture}_{suffix}",
                                'train', f'best_weights_fold2.tar')
    else:
        model_dir = os.path.join(root_path, 'results', version, 
                                f"{config['patient']}_{config['data_type']}_{model_architecture}_{suffix}",
                                'train', f'model_weights_epoch{config["epoch"]}.tar')
    
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    print(torch.load(model_dir)['epoch'])
    model.eval()

    predictions_all = np.empty((0, args['num_labels']))
    predictions_length = {}
    all_attentions = defaultdict(list)  # For collecting attentions per layer
    with torch.no_grad():
        if config['patient'] == 'p10' and '1' in phase and 'CR' not in phase:
            for ph in ['FR1a', 'FR1b']:
                predictions = np.empty((0, args['num_labels']))
                args['free_recall_phase'] = ph
                dataloaders = initialize_inference_dataloaders(args)
                for i, (feature, index) in enumerate(dataloaders['inference']):
                    if not args['use_lfp'] and args['use_spike']:
                        spike = feature.to(args['device'])
                        lfp = None
                    elif args['use_lfp'] and not args['use_spike']:
                        lfp = feature.to(args['device'])
                        spike = None
                    else:
                        assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                        spike = feature[1].to(args['device'])
                        lfp = feature[0].to(args['device'])
                    spike_emb, lfp_emb, output, attentions = model(lfp, spike)
                    output = torch.sigmoid(output)
                    pred = output.cpu().detach().numpy()
                    predictions = np.concatenate([predictions, pred], axis=0)
                    # Collect attentions
                    for layer_idx, attn in enumerate(attentions):
                        all_attentions[layer_idx].append(attn.cpu().numpy())

                if args['use_overlap']:
                    fake_activation = np.mean(predictions, axis=0)
                    predictions = np.vstack((fake_activation, predictions, fake_activation))
                    
                predictions_all = np.concatenate([predictions_all, predictions], axis=0)
            predictions_length[phase] = len(predictions_all)
        else:
            args['free_recall_phase'] = phase
            dataloaders = initialize_inference_dataloaders(args)    
            predictions = np.empty((0, args['num_labels']))
            for i, (feature, index) in enumerate(dataloaders['inference']):
                if not args['use_lfp'] and args['use_spike']:
                    spike = feature.to(args['device'])
                    lfp = None
                elif args['use_lfp'] and not args['use_spike']:
                    lfp = feature.to(args['device'])
                    spike = None
                else:
                    assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                    spike = feature[1].to(args['device'])
                    lfp = feature[0].to(args['device'])
                spike_emb, lfp_emb, output, attentions = model(lfp, spike)
                output = torch.sigmoid(output)
                pred = output.cpu().detach().numpy()
                predictions = np.concatenate([predictions, pred], axis=0)
                # Collect attentions
                for layer_idx, attn in enumerate(attentions):
                    all_attentions[layer_idx].append(attn.cpu().numpy())
            
            if args['use_overlap']:
                fake_activation = np.mean(predictions, axis=0)
                predictions = np.vstack((fake_activation, predictions, fake_activation))

            predictions_length[phase] = len(predictions)
            predictions_all = np.concatenate([predictions_all, predictions], axis=0)

    for ph in alongwith:
        args['free_recall_phase'] = ph
        dataloaders = initialize_inference_dataloaders(args)
        with torch.no_grad():
            predictions = np.empty((0, args['num_labels']))
            for i, (feature, index) in enumerate(dataloaders['inference']):
                if not args['use_lfp'] and args['use_spike']:
                    spike = feature.to(args['device'])
                    lfp = None
                elif args['use_lfp'] and not args['use_spike']:
                    lfp = feature.to(args['device'])
                    spike = None
                else:
                    assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                    spike = feature[1].to(args['device'])
                    lfp = feature[0].to(args['device'])

                spike_emb, lfp_emb, output, attentions = model(lfp, spike)
                output = torch.sigmoid(output)
                pred = output.cpu().detach().numpy()
                predictions = np.concatenate([predictions, pred], axis=0)
                # Collect attentions
                for layer_idx, attn in enumerate(attentions):
                    all_attentions[layer_idx].append(attn.cpu().numpy())
            
            if args['use_overlap']:
                fake_activation = np.mean(predictions, axis=0)
                predictions = np.vstack((fake_activation, predictions, fake_activation))

        predictions_length[ph] = len(predictions)
        predictions_all = np.concatenate([predictions_all, predictions], axis=0)

    smoothed_data = np.zeros_like(predictions_all)
    for i in range(predictions_all.shape[1]):
        smoothed_data[:, i] = np.convolve(predictions_all[:, i], np.ones(4)/4, mode='same')
    predictions = predictions_all
    
    # Run requested visualizations
    save_path = os.path.join(root_path, 'results', version, 
                            f"{config['patient']}_{config['data_type']}_{model_architecture}_{suffix}",
                            'memory', config['data_version'], f'epoch{config["epoch"]}_{phase}_{len(alongwith)}')
    os.makedirs(save_path, exist_ok=True)

    if save_predictions:
        np.save(os.path.join(save_path, 'free_recall_predictions.npy'), predictions_all)

    # window_size, start, end, step_size = 4, -12, 12, 1
    # windows = [[i, i + window_size] for i in range(start, end - window_size + 1, step_size)]
    # # windows = [[-4, 0]]
    # for window in windows:
    #     window_name = str((window[0] + window[1]) // 2)
    #     save_path = os.path.join(root_path, 'results', version, 
    #                         f"{config['patient']}_{config['data_type']}_{model_architecture}_{suffix}",
    #                         'memory', config['data_version'], f'epoch{config["epoch"]}_{phase}_{len(alongwith)}', window_name)
    #     os.makedirs(save_path, exist_ok=True)

    method_MCS(
        smoothed_data, config['patient'], phase, save_path,
        use_clusterless=args['use_spike'],
        use_lfp=args['use_lfp'],
        use_combined=args['use_combined'],
        alongwith=alongwith,
        predictions_length=predictions_length,
        # window=window
    )
    # method_heatmap(predictions, config['patient'], phase, save_path)

    # Figure 2c
    # method_curve_shape(
    #     smoothed_data, config['patient'], phase, save_path,
    #     use_clusterless=args['use_spike'],
    #     use_lfp=args['use_lfp'],
    #     use_combined=args['use_combined'],
    #     alongwith=alongwith,
    #     predictions_length=predictions_length
    # )

    # method_curve_shape_all(
    #     smoothed_data, config['patient'], phase, save_path,
    #     use_clusterless=args['use_spike'],
    #     use_lfp=args['use_lfp'],
    #     use_combined=args['use_combined'],
    #     alongwith=alongwith,
    #     predictions_length=predictions_length
    # )

def create_test_configs(train_configs):
    """Create test configurations for both data versions and multiple epochs."""
    test_configs = []
    for config in train_configs:
        for data_version in ['nature-setting-precise', 'nature-setting-est']:
            for epoch in config['save_epochs']:
                for patient in config['patient']:  # Iterate through each patient in the list
                    test_config = config.copy()
                    test_config['data_version'] = data_version
                    test_config['epoch'] = epoch
                    test_config['patient'] = patient  # Set single patient
                    test_configs.append(test_config)
    return test_configs

# Utility for lesion suffix

def get_lesion_suffix(lesion, lesion_mode):
    return f'{lesion_mode}_{lesion}' if lesion != 'Full' else 'Full'

# Utility for version string

def get_version_string(num_csa_layers, num_cca_layers, hidden_size, num_attention_heads):
    return f"CSA{num_csa_layers}_CCA{num_cca_layers}_{hidden_size}_H{num_attention_heads}"

# Utility for checkpoint path

def get_ckpt_path(label_version, participant, suffix, epoch):
    return f'results/{label_version}/{participant}_clusterless_transformer_{suffix}/train/model_weights_epoch{epoch}.tar'

def main():
    # List of all 10 patients
    all_patients = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
    # Patient exclusions for specific lesion configs
    patient_exclusions = {
        ('MTL', 'without'): ['p4'],
        ('FC', 'only'): ['p4'],
        ('HPC', 'only'): ['p9'],
    }
    train_configs = create_configs()
    root_path = './'

    # Group configs by architecture parameters
    arch_groups = {}
    for config in train_configs:
        arch_key = (
            config['num_csa_layers'],
            config['num_cca_layers'],
            config['hidden_size'],
            config['num_attention_heads']
        )
        if arch_key not in arch_groups:
            arch_groups[arch_key] = []
        arch_groups[arch_key].append(config)

    for arch_key, configs in arch_groups.items():
        print(f"\nProcessing architecture: CSA={arch_key[0]}, CCA={arch_key[1]}, Hidden={arch_key[2]}, Heads={arch_key[3]}")
        # Build a dict: lesion_type -> list of configs (one per patient), and keep patient list per lesion
        lesion_dict = {'Full': [], 'MTL': [], 'FC': [], 'HPC': []}
        patients_per_lesion = {'Full': [], 'MTL': [], 'FC': [], 'HPC': []}
        for config in configs:
            lesion_key = config['lesion'] if config['lesion'] else 'Full'
            if lesion_key in lesion_dict:
                # Apply patient_exclusions for this config
                excluded = []
                if (config['lesion'], config['lesion_mode']) in patient_exclusions:
                    excluded = patient_exclusions[(config['lesion'], config['lesion_mode'])]
                for patient in config['patient']:
                    if patient in all_patients and patient not in excluded:
                        c = config.copy()
                        c['patient'] = patient
                        lesion_dict[lesion_key].append(c)
                        if patient not in patients_per_lesion[lesion_key]:
                            patients_per_lesion[lesion_key].append(patient)

        # 1. Train all patients for each lesion type
        for lesion in ['Full', 'MTL', 'FC', 'HPC']:
            print(f"\nTraining all patients for lesion={lesion}...")
            for config in lesion_dict[lesion]:
                # For 'Full', always use '_Full' as the suffix
                # For MTL/FC, use _{lesion_mode}_{lesion} (e.g., _only_MTL, _without_FC)
                lesion_suffix = get_lesion_suffix(config["lesion"], config["lesion_mode"])
                suffix = lesion_suffix
                version = get_version_string(config['num_csa_layers'], config['num_cca_layers'], config['hidden_size'], config['num_attention_heads'])
                data_type = config['data_type']
                use_clusterless, use_lfp, use_combined, model_architecture = get_model_flags(data_type)
                save_paths = build_save_paths(root_path, version, config['patient'], data_type, model_architecture, suffix)
                args = build_args(config, config['patient'], use_clusterless, use_lfp, use_combined, model_architecture, save_paths)
                trainer = pipeline(args)
                trainer.train(args['epochs'], 1)
                
        # 2. Test all patients for each lesion type, all epochs, both data versions
        for lesion in ['Full', 'MTL', 'FC', 'HPC']:
            print(f"\nTesting all patients for lesion={lesion}...")
            for config in lesion_dict[lesion]:
                for data_version in ['simulated']:
                    for epoch in config['save_epochs']:
                        test_config = config.copy()
                        test_config['data_version'] = data_version
                        test_config['epoch'] = epoch
                        test_config['patient'] = config['patient']
                        print(f"Testing model for patient {config['patient']}, lesion={lesion}, data_version={data_version}, epoch={epoch}...")
                        if config['patient'] in ['p1', 'p2']:
                            perform_memory_test(test_config, phase='FR1', save_predictions=True)
                            perform_memory_test(test_config, phase='FR2', save_predictions=True)
                        else:
                            perform_memory_test(test_config, phase='FR1', alongwith=['CR1'], save_predictions=True)
                            perform_memory_test(test_config, phase='FR2', alongwith=['CR2'], save_predictions=True)

if __name__ == '__main__':
    main() 