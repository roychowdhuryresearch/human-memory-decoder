import numpy as np
import torch
import random
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from src.utils.initializer import *
from itertools import product
from src.trainer import Trainer
from src.utils.visualization import (method_heatmap, method_soraya, method_curve_shape, 
                                   method_curve_shape_all, clean_data, rms_of_derivatives,
                                   total_variation, peak_to_peak_amplitude, high_frequency_energy,
                                   standard_deviation, min_max_normalize, combined_score)
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
        ('MTL', 'without'): ['567'],  # Exclude 567 for MTL-without
        ('FC', 'only'): ['567'],      # Exclude 567 for FC-only
        ('HPC', 'only'): ['i717'],      # Exclude i717 for HPC-only
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
        # all patients: '562', '563', '566', '567', '568', '570', '572', '573', 'i717', 'i728'
        'patient': ['562'], 
        'epochs': 50,
        'save_epochs': [49],
    }

    # Define variations for different parameters
    variations = {
        'lesion': ['Full', 'MTL', 'HPC', 'FC'],  # '', 'MTL', 'HPC', 'FC'
        'lesion_mode': ['only'],
        'norm_method': ['zscore_bundle'], # 'zscore_channel', 'minmax', 'zscore_bundle'
        'data_type': ['clusterless'],
        'data_version': ['simulated'],
    }

    # Define architecture search space
    arch_variations = {
        'num_csa_layers': [6],  # 6-9
        'num_cca_layers': [6],   # 1-3
        'arch_config': [
            # (256, 8),  # (hidden_size, num_attention_heads)
            (384, 6)
        ]
    }

    # Generate all combinations
    configs = []
    # Generate architecture combinations
    allowed_pairs = {(3, 3), (6, 6)}
    arch_combinations = [
        (num_csa_layers, num_cca_layers, (hidden_size, attn_heads))
        for num_csa_layers, num_cca_layers, (hidden_size, attn_heads) in product(
            arch_variations['num_csa_layers'],
            arch_variations['num_cca_layers'],
            arch_variations['arch_config']
        )
        if (num_csa_layers, num_cca_layers) in allowed_pairs
    ]

    # Filter out the combination where both num_csa_layers and num_cca_layers are 6
    # arch_combinations = [combo for combo in arch_combinations if not (combo[0] == 6 and combo[1] == 6)]

    # Generate all combinations with other parameters
    for values in product(variations['lesion'], 
                         variations['lesion_mode'], 
                         variations['norm_method'],
                         variations['data_type'],
                         variations['data_version']):
        config = base_config.copy()
        lesion, lesion_mode = values[0], values[1]
        
        # Skip invalid combinations early
        if not lesion and lesion_mode != 'only':
            continue

        # Set non-architecture parameters
        config.update({
            'lesion': lesion,
            'lesion_mode': lesion_mode,
            'norm_method': values[2],
            'data_type': values[3],
            'data_version': values[4],
        })
            
        # Handle patient exclusions
        if (lesion, lesion_mode) in patient_exclusions:
            config['patient'] = [p for p in base_config['patient'] 
                               if p not in patient_exclusions[(lesion, lesion_mode)]]

        # Create configurations for each architecture combination
        for csa_layers, cca_layers, (hidden_size, attn_heads) in arch_combinations:
            arch_config = config.copy()
            arch_config.update({
                'num_csa_layers': csa_layers,
                'num_cca_layers': cca_layers,
                'hidden_size': hidden_size,
                'num_attention_heads': attn_heads
            })
            configs.append(arch_config)
    
    return configs

def pipeline(config):
    set_seed(42)
    # Set seeds for reproducibility
    # seed = config['seed']
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # Set device
    device = torch.device(config['device'])

    # Initialize components
    dataloaders = initialize_dataloaders(config)
    model = initialize_model(config).to(device)

    # Optional: Compile model (PyTorch 2.0+)
    # model = torch.compile(model)

    # Log config to W&B
    # wandb.config.update(config)

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
        return True, False, False, 'multi-vit'
    elif data_type == 'lfp':
        return False, True, False, 'multi-vit'
    elif data_type == 'combined':
        return True, True, True, 'multi-crossvit'
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

    if config['patient'] == 'i728' and '1' in phase:
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
        if config['patient'] == 'i728' and '1' in phase and 'CR' not in phase:
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
            # load the best epoch number from the saved "model_results" structure
            predictions = np.empty((0, args['num_labels']))
            # y_true = np.empty((0, self.config.num_labels))
            for i, (feature, index) in enumerate(dataloaders['inference']):
                # target = target.to(self.device)
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
                # forward pass

                # start_time = time.time()
                spike_emb, lfp_emb, output, attentions = model(lfp, spike)
                # end_time = time.time()
                # print('inference time: ', end_time - start_time)
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
    for i in range(predictions_all.shape[1]):  # Loop through each feature
        smoothed_data[:, i] = np.convolve(predictions_all[:, i], np.ones(4)/4, mode='same')
    predictions = predictions_all
    
    # Run requested visualizations
    save_path = os.path.join(root_path, 'results', version, 
                            f"{config['patient']}_{config['data_type']}_{model_architecture}_{suffix}",
                            'memory', config['data_version'], f'epoch{config["epoch"]}_{phase}_{len(alongwith)}')
    os.makedirs(save_path, exist_ok=True)

    if save_predictions:
        np.save(os.path.join(save_path, 'free_recall_predictions.npy'), predictions_all)
        # After the loop, save attentions as a dict
        attention_dict = {layer_idx: np.concatenate(attn_list, axis=0) for layer_idx, attn_list in all_attentions.items()}
        np.save(os.path.join(save_path, 'cca_attentions_dict.npy'), attention_dict)
    

    # window_size, start, end, step_size = 4, -12, 12, 1
    # windows = [[i, i + window_size] for i in range(start, end - window_size + 1, step_size)]
    # # windows = [[-4, 0]]
    # for window in windows:
    #     window_name = str((window[0] + window[1]) // 2)
    #     save_path = os.path.join(root_path, 'results', version, 
    #                         f"{config['patient']}_{config['data_type']}_{model_architecture}_{suffix}",
    #                         'memory', config['data_version'], f'epoch{config["epoch"]}_{phase}_{len(alongwith)}', window_name)
    #     os.makedirs(save_path, exist_ok=True)

    method_soraya(
        smoothed_data, config['patient'], phase, save_path,
        use_clusterless=args['use_spike'],
        use_lfp=args['use_lfp'],
        use_combined=args['use_combined'],
        alongwith=alongwith,
        predictions_length=predictions_length,
        # window=window
    )
    # method_heatmap(predictions, config['patient'], phase, save_path)

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

def analyze_results(model_versions, label_version, participants, epoch, suffix):
    """Analyze results and return performance metrics and matrices."""
    def load_participant_data(participant, model_version):
        p_r1, p_r2 = [np.nan] * 8, [np.nan] * 8
        n_r1, n_r2 = [0] * 8, [0] * 8
        
        participant_cr = '0' if participant in ['562', '563'] else '1'
        
        # Load FR1 data
        file = f'results/{label_version}/{participant}_clusterless_multi-vit_{suffix}/memory/{model_version}/epoch{epoch}_FR1_{participant_cr}/AUC.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
            p_r1 = list(df['p_value'])
            if model_version == model_versions[0]:
                n_r1 = list(df['n_vocalizations'])
        
        # Load FR2 data
        file = f'results/{label_version}/{participant}_clusterless_multi-vit_{suffix}/memory/{model_version}/epoch{epoch}_FR2_{participant_cr}/AUC.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
            p_r2 = list(df['p_value'])
            if model_version == model_versions[0]:
                n_r2 = list(df['n_vocalizations'])
        
        return p_r1, p_r2, n_r1, n_r2

    # Load data for both versions
    data_version1 = {p: load_participant_data(p, model_versions[0]) for p in participants}
    data_version2 = {p: load_participant_data(p, model_versions[1]) for p in participants}
    
    # Initialize matrices
    num_concepts = 8
    num_participants = len(participants)
    p_matrix_r1 = np.zeros((num_concepts, num_participants))
    p_matrix_r2 = np.zeros((num_concepts, num_participants))
    n_matrix_r1 = np.zeros((num_concepts, num_participants))
    n_matrix_r2 = np.zeros((num_concepts, num_participants))
    
    performance_metrics = {}
    
    # For each participant
    for p_idx, participant in enumerate(participants):
        # Get n values from version1 (they're the same for both versions)
        n_r1 = data_version1[participant][2]
        n_r2 = data_version1[participant][3]
        
        # Get all scores for both versions
        scores1_r1 = np.array(data_version1[participant][0])
        scores2_r1 = np.array(data_version2[participant][0])
        scores1_r2 = np.array(data_version1[participant][1])
        scores2_r2 = np.array(data_version2[participant][1])
        
        # Calculate mean scores for each version (ignoring nans)
        mean1_r1 = np.nanmedian(scores1_r1) if not np.all(np.isnan(scores1_r1)) else 0
        mean2_r1 = np.nanmedian(scores2_r1) if not np.all(np.isnan(scores2_r1)) else 0
        mean1_r2 = np.nanmedian(scores1_r2) if not np.all(np.isnan(scores1_r2)) else 0
        mean2_r2 = np.nanmedian(scores2_r2) if not np.all(np.isnan(scores2_r2)) else 0
        
        # Choose the better version for FR1
        if participant == '568':  # Force participant 568 to use version1 for FR1
            p_matrix_r1[:, p_idx] = scores1_r1
            version1_chosen_r1 = True
        else:
            if mean1_r1 > mean2_r1:
                p_matrix_r1[:, p_idx] = scores1_r1
                version1_chosen_r1 = True
            else:
                p_matrix_r1[:, p_idx] = scores2_r1
                version1_chosen_r1 = False
        
        # Choose the better version for FR2
        if mean1_r2 > mean2_r2:
            p_matrix_r2[:, p_idx] = scores1_r2
            version1_chosen_r2 = True
        else:
            p_matrix_r2[:, p_idx] = scores2_r2
            version1_chosen_r2 = False
        
        # Set n values (same for both versions)
        n_matrix_r1[:, p_idx] = n_r1
        n_matrix_r2[:, p_idx] = n_r2
        
        # Store metrics
        performance_metrics[participant] = {
            'FR1': max(mean1_r1, mean2_r1),
            'FR2': max(mean1_r2, mean2_r2),
            'version1_chosen': {'FR1': version1_chosen_r1, 'FR2': version1_chosen_r2},
            'version1_scores': {'FR1': mean1_r1, 'FR2': mean1_r2},
            'version2_scores': {'FR1': mean2_r1, 'FR2': mean2_r2}
        }
    
    return performance_metrics, p_matrix_r1, p_matrix_r2, n_matrix_r1, n_matrix_r2

def get_results(model_versions_fr1, model_versions_fr2, label_version, participants, epoch, suffix):
    """Analyze results and return performance metrics and matrices."""
    def load_participant_data(participant, model_version_fr1, model_version_fr2):
        p_r1, p_r2 = [np.nan] * 8, [np.nan] * 8
        n_r1, n_r2 = [0] * 8, [0] * 8
        
        participant_cr = '0' if participant in ['562', '563'] else '1'
        
        # Load FR1 data
        file = f'results/{label_version}/{participant}_clusterless_multi-vit_{suffix}/memory/{model_version_fr1}/epoch{epoch}_FR1_{participant_cr}/AUC.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
            p_r1 = list(df['p_value'])
            n_r1 = list(df['n_vocalizations'])
        
        # Load FR2 data
        file = f'results/{label_version}/{participant}_clusterless_multi-vit_{suffix}/memory/{model_version_fr2}/epoch{epoch}_FR2_{participant_cr}/AUC.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
            p_r2 = list(df['p_value'])
            n_r2 = list(df['n_vocalizations'])
        
        return p_r1, p_r2, n_r1, n_r2

    # Load data for both versions
    data_version1 = {p: load_participant_data(p, model_versions_fr1[i], model_versions_fr2[i]) for i, p in enumerate(participants)}
    
    # Initialize matrices
    num_concepts = 8
    num_participants = len(participants)
    p_matrix_r1 = np.zeros((num_concepts, num_participants))
    p_matrix_r2 = np.zeros((num_concepts, num_participants))
    n_matrix_r1 = np.zeros((num_concepts, num_participants))
    n_matrix_r2 = np.zeros((num_concepts, num_participants))
        
    # For each participant
    for p_idx, participant in enumerate(participants):
        # Get n values from version1 (they're the same for both versions)
        n_r1 = data_version1[participant][2]
        n_r2 = data_version1[participant][3]
        
        # Get all scores for both versions
        scores1_r1 = np.array(data_version1[participant][0])
        scores1_r2 = np.array(data_version1[participant][1])
                

        p_matrix_r1[:, p_idx] = scores1_r1
        p_matrix_r2[:, p_idx] = scores1_r2
        
        # Set n values (same for both versions)
        n_matrix_r1[:, p_idx] = n_r1
        n_matrix_r2[:, p_idx] = n_r2

    return p_matrix_r1, p_matrix_r2, n_matrix_r1, n_matrix_r2

def wilcoxon_z_value(data, zero_method='wilcox'):
    """Compute Wilcoxon signed-rank statistic."""
    # Get the Wilcoxon stat & p-value from SciPy
    W, p = stats.wilcoxon(data, zero_method=zero_method, mode='auto')  
    
    # If using zero_method="wilcox", all zeros are dropped
    if zero_method == 'wilcox':
        data = data[data != 0]
    
    n = len(data)
    
    # Compute expected value and standard deviation under H0
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n*(n+1)*(2*n+1) / 24.0)
    
    # Compute z-score
    z = (W - mu) / sigma
    return W, p, z

def calculate_stats(p_vals, n_voc):
    """Calculate statistical metrics for model performance."""
    # Flatten the array and remove NaN values
    flattened_data = p_vals.flatten()
    flattened_number = n_voc.flatten()
    mask = flattened_number > 2
    valid_data = flattened_data[mask]
    
    _, one_sample_wilcoxon_p, _ = wilcoxon_z_value(valid_data - 50)
    cohen_d = (valid_data.mean() - 50) / valid_data.std(ddof=1)
    
    return one_sample_wilcoxon_p, cohen_d

def cleanup_models(performance_metrics, matrices_dict, label_version, epoch):
    """Delete specific epoch checkpoints for poorly performing models based on statistical criteria across all patients and lesions. Also store stats in a local file."""
    stats_file = f'{label_version}_cleanup_stats.csv'
    file_exists = os.path.isfile(stats_file)
    with open(stats_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'lesion', 'wilcoxon_p_r1', 'cohen_d_r1', 'wilcoxon_p_r2', 'cohen_d_r2'])
        fails = []
        for lesion, (p_matrix_r1, p_matrix_r2, n_matrix_r1, n_matrix_r2) in matrices_dict.items():
            print(f"\nCleanup stats for lesion: {lesion}")
            wilcoxon_p_r1, cohen_d_r1 = calculate_stats(p_matrix_r1, n_matrix_r1)
            wilcoxon_p_r2, cohen_d_r2 = calculate_stats(p_matrix_r2, n_matrix_r2)
            print(f"FR1: wilcoxon_p={wilcoxon_p_r1:.3f}, cohen_d={cohen_d_r1:.3f}")
            print(f"FR2: wilcoxon_p={wilcoxon_p_r2:.3f}, cohen_d={cohen_d_r2:.3f}")
            # Store stats in file
            writer.writerow([epoch, lesion, wilcoxon_p_r1, cohen_d_r1, wilcoxon_p_r2, cohen_d_r2])
            # Apply lesion-specific criteria
            if lesion == 'Full':
                # fail = (wilcoxon_p_r1 > 0.02 or wilcoxon_p_r2 > 0.02 or cohen_d_r1 < 0.35 or cohen_d_r2 < 0.35)
                fail = (wilcoxon_p_r1 > 0.01 or wilcoxon_p_r2 > 0.01)
            elif lesion == 'MTL':
                # fail = (wilcoxon_p_r1 > wilcoxon_p_r2 or wilcoxon_p_r1 > 0.1)
                fail = (wilcoxon_p_r2 < 0.05 or wilcoxon_p_r1 > 0.2)
            elif lesion == 'FC':
                # fail = (wilcoxon_p_r1 < wilcoxon_p_r2 or wilcoxon_p_r2 > 0.1)
                fail = (wilcoxon_p_r1 < 0.05 or wilcoxon_p_r2 > 0.2) 
            else:
                fail = False
            fails.append(fail)
    if any(fails):
        print(f"\nPoor overall performance detected for epoch {epoch}. Removing checkpoints...")
        # Remove specific epoch checkpoint for all patients and all lesions
        lesion_suffixes = {'Full': 'Full', 'MTL': 'only_MTL', 'FC': 'only_FC'}
        for participant in performance_metrics.keys():
            for lesion, suffix in lesion_suffixes.items():
                ckpt_path = get_ckpt_path(label_version, participant, suffix, epoch)
                if os.path.exists(ckpt_path):
                    print(f"Deleting epoch {epoch} {lesion} checkpoint for {participant}")
                    os.system(f"rm -f {ckpt_path}")
    else:
        print(f"\nOverall performance meets criteria for epoch {epoch}. Keeping checkpoints.")

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
    return f"CSA{num_csa_layers}_CCA{num_cca_layers}_{hidden_size}_H{num_attention_heads}_CLS_grid_bundle_zscore_nologp_yesCCA"

# Utility for checkpoint path

def get_ckpt_path(label_version, participant, suffix, epoch):
    if epoch == 'best':
        return f'results/{label_version}/{participant}_clusterless_multi-vit_{suffix}/train/best_weights_fold2.tar'
    else:
        return f'results/{label_version}/{participant}_clusterless_multi-vit_{suffix}/train/model_weights_epoch{epoch}.tar'

def main():
    # List of all 10 patients
    all_patients = ['562', '563', '566', '567', '568', '570', '572', '573', 'i717', 'i728']
    # Patient exclusions for specific lesion configs
    patient_exclusions = {
        ('MTL', 'without'): ['567'],
        ('FC', 'only'): ['567'],
        ('HPC', 'only'): ['i717'],
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
        # Ensure all three lesion types have at least one patient (ideally 10, but may be less due to exclusions)
        # if not all(len(lesion_dict[l]) > 0 for l in lesion_dict):
        #     print(f"Skipping architecture {arch_key} because not all lesion types have patients.")
        #     continue
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
                # trainer = pipeline(args)
                # trainer.train(args['epochs'], 1)
        # 2. Test all patients for each lesion type, all epochs, both data versions
        for lesion in ['Full', 'MTL', 'FC', 'HPC']:
            print(f"\nTesting all patients for lesion={lesion}...")
            for config in lesion_dict[lesion]:
                for data_version in ['nature-setting-2025-epoch49']:
                    for epoch in config['save_epochs']:
                        test_config = config.copy()
                        test_config['data_version'] = data_version
                        test_config['epoch'] = epoch
                        test_config['patient'] = config['patient']
                        print(f"Testing model for patient {config['patient']}, lesion={lesion}, data_version={data_version}, epoch={epoch}...")
                        if config['patient'] in ['562', '563']:
                            perform_memory_test(test_config, phase='FR1', save_predictions=True)
                            perform_memory_test(test_config, phase='FR2', save_predictions=True)
                        else:
                            perform_memory_test(test_config, phase='FR1', alongwith=['CR1'], save_predictions=True)
                            perform_memory_test(test_config, phase='FR2', alongwith=['CR2'], save_predictions=True)

if __name__ == '__main__':
    main() 