import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
from scipy import stats
import ast
from .free_recall_time import *
from src.param.param_data import LABELS8

# Helper functions for curve analysis
def rms_of_derivatives(curve):
    """Calculate Root Mean Square of derivatives."""
    derivative = np.diff(curve)
    rms = np.sqrt(np.mean(derivative**2))
    return rms

def total_variation(curve):
    """Calculate total variation of the curve."""
    return np.sum(np.abs(np.diff(curve)))

def peak_to_peak_amplitude(curve):
    """Calculate peak-to-peak amplitude of the curve."""
    return np.max(curve) - np.min(curve)

def high_frequency_energy(curve):
    """Calculate sum of high-frequency components using Fourier Transform."""
    fft = np.fft.fft(curve)
    frequencies = np.fft.fftfreq(len(curve))
    high_freq_energy = np.sum(np.abs(fft[np.abs(frequencies) > 0.1])**2)
    return high_freq_energy

def standard_deviation(curve):
    """Calculate standard deviation of the curve."""
    return np.std(curve)

def min_max_normalize(value, min_value, max_value):
    """Normalize value to range [0,1] using min-max scaling."""
    return (value - min_value) / (max_value - min_value)

def combined_score(normalized_metrics):
    """Calculate combined score from normalized metrics."""
    return sum(normalized_metrics[metric] for metric in normalized_metrics) / len(normalized_metrics)

def clean_data(data):
    """Clean data by handling string representations of lists and None values."""
    return [
        ast.literal_eval(item) if isinstance(item, str) else np.nan if item is None or np.isnan(item) else item
        for item in data
    ]

def method_heatmap(predictions, patient, phase, save_path):
    """Generate heatmap visualization of predictions.
    
    Args:
        predictions: Model predictions array
        patient: Patient ID
        phase: Test phase
        save_path: Path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(4, 8))
    heatmap = ax.imshow(predictions, cmap='viridis', aspect='auto', interpolation='none')

    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=10)
    tick_positions = np.arange(0, len(predictions), 15*4)  # 15 seconds * 100 samples per second
    tick_labels = [int(pos * 0.25) for pos in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.arange(0, predictions.shape[1], 1))
    ax.set_xticklabels(LABELS8, rotation=80)
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Concept')
    plt.title(f'{patient} {phase} predictions')
    plt.tight_layout()
    file_path = os.path.join(save_path, f'free_recall_activations_{phase}.png')
    plt.savefig(file_path)
    plt.close()

def method_MCS(predictions, patient, phase, save_path, use_clusterless=False, use_lfp=False, use_combined=False, alongwith=[], predictions_length={}):
    """Generate Soraya's method visualization - speed optimized version with identical logic."""
    result_df = pd.DataFrame()
    min_vocalizations = 2
    p_stats = {}
    s_stats = {}
    CR_bins = []
    
    # Pre-compute window data (same logic as original)
    if 'FR' in phase and any('CR' in element for element in alongwith):
        free_recall_windows_fr = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        free_recall_windows_cr = eval('free_recall_windows' + '_' + patient + f'_{alongwith[0]}')
        surrogate_windows_fr = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        surrogate_windows_cr = eval('surrogate_windows' + '_' + patient + f'_{alongwith[0]}')
        viewing_windows_cr = eval('viewing_windows' + '_' + patient + f'_{alongwith[0]}')
        offset = int(predictions_length[phase] * 0.25) * 1000
        viewing_bins = [np.arange(max(s, 0), max(e, 0), 0.25) * 1000 + offset for (s, e) in viewing_windows_cr]
        viewing_bins = np.concatenate(viewing_bins)
        CR_bins = [predictions_length[phase], predictions_length[phase] + predictions_length[alongwith[0]]]
        free_recall_windows = [fr + [cr_item + offset for cr_item in cr]  for fr, cr in zip(free_recall_windows_fr, free_recall_windows_cr)]
        surrogate_windows = surrogate_windows_fr + [cr_item + offset for cr_item in surrogate_windows_cr]
    elif 'FR' in phase and not any('CR' in element for element in alongwith):
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        viewing_bins = np.array([])
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        viewing_bins = np.array([])

    # Window merging (same logic as original)
    temp = []
    la = free_recall_windows[0]
    ba = free_recall_windows[1]
    wh = free_recall_windows[2]
    cia = free_recall_windows[3]
    hostage = free_recall_windows[4]
    handcuff = free_recall_windows[5]
    jack = free_recall_windows[6]
    chloe = free_recall_windows[7]
    bill = free_recall_windows[8]
    fayed = free_recall_windows[9]
    amar = free_recall_windows[10]
    president = free_recall_windows[11]
    
    # merge whiltehouse and president
    whitehouse = wh + president
    # merge CIA and Chloe
    CIA = cia + chloe
    
    temp.append(whitehouse)
    temp.append(CIA)
    temp.append(hostage)
    temp.append(handcuff)
    temp.append(jack)
    temp.append(bill)
    temp.append(fayed)
    temp.append(amar)
    free_recall_windows = temp

    # Pre-compute analysis parameters (moved outside loop for speed)
    analysis_params = {
        "bin_size": 0.25,  # seconds
        "win_range_sec": [-4, 0],
        "rand_trial_separation_sec": 4,  # seconds
        "activation_threshold": 0,
        "threshold_type": "mean",
        "penalize_sub_threshold": True,
        "n_permutations": 1500,
    }
    
    activations = predictions
    n_permutations = analysis_params['n_permutations']
    bin_size = analysis_params['bin_size']
    win_range_sec = analysis_params['win_range_sec']
    rand_trial_separation_sec = analysis_params['rand_trial_separation_sec']

    # Pre-compute time vector and window bins (moved outside loop for speed)
    time = np.arange(0, activations.shape[0], 1) * bin_size
    win_range_bins = [int(x / bin_size) for x in win_range_sec]
    rand_trial_separation_bins = rand_trial_separation_sec / bin_size

    # Pre-compute viewing bins mask if needed (moved outside loop for speed)
    if viewing_bins.size > 0:
        ignore_bin = np.array([np.abs(time - bin_val / 1000).argmin() for bin_val in viewing_bins])
    else:
        ignore_bin = np.array([])

    for n_concept in range(len(LABELS8)):
        p_values = []
        s_scores = []
        all_voc_scores = []
        all_surro_scores = []
        
        concept = LABELS8[n_concept]
        n_bins = np.abs(win_range_bins[1] - win_range_bins[0]) + 1
        activation = activations[:, n_concept]

        # Create mask for threshold calculation (same logic as original)
        cr_mask = np.ones(activation.shape[0], dtype=bool)
        if ignore_bin.size > 0:
            cr_mask[ignore_bin] = False
        thresh = np.mean(activation[cr_mask])

        concept_vocalz_msec = free_recall_windows[n_concept]
        n_vocalizations = len(concept_vocalz_msec)
        n_rand_trials = n_vocalizations

        if n_vocalizations <= min_vocalizations:  # skip if no vocalizations for this concept
            p_values.append(np.nan)
            s_scores.append(np.nan)
            all_voc_scores.append(np.nan)
            all_surro_scores.append(np.nan)
        else:
            _, target_activations_indices = find_target_activation_indices(
                time, concept_vocalz_msec, win_range_bins, end_inclusive=True
            )
            voc_activations = []
            for activation_indices in target_activations_indices:
                voc_activations.append(activation[activation_indices])

            if len(voc_activations) == 0:
                p_values.append(np.nan)
                s_scores.append(np.nan)
                all_voc_scores.append(np.nan)
                all_surro_scores.append(np.nan)
                continue

            # SPEED OPTIMIZATION: Vectorized AUC calculation
            voc_auc = []
            voc_activations_array = np.array(voc_activations)
            adjusted_acts = np.clip(voc_activations_array - thresh, a_min=0, a_max=None)
            voc_auc = np.trapz(adjusted_acts, axis=1)  # Vectorized trapezoidal integration
            
            mean_voc_act = np.mean(voc_activations, 0)
            mean_voc_auc = np.mean(voc_auc)
            mean_voc_act[mean_voc_act < thresh] = thresh

            # determine mean of AUC for surrogate "trials" (same logic as original)
            surrogate_vocalz_msec = [
                s for s in surrogate_windows 
                if all(not (cvm + win_range_sec[0] * 1000 < s <= cvm) for cvm in concept_vocalz_msec)
            ]
            
            _, surrogate_indices = find_target_activation_indices(
                time, surrogate_vocalz_msec, win_range_bins, end_inclusive=True
            )
            surrogate_bins = np.array(surrogate_indices)
            n_surrogate_vocalizations = len(surrogate_bins)

            np.random.seed(42)
            
            # Generate all random indices first (for loop only for this)
            all_random_trial_indices = []
            for i in range(n_permutations):
                random_trial_indices = np.random.choice(n_surrogate_vocalizations, n_rand_trials, replace=False)
                all_random_trial_indices.append(random_trial_indices)
            
            # Convert to numpy array for vectorized operations
            all_random_trial_indices = np.array(all_random_trial_indices)  # Shape: (n_permutations, n_rand_trials)
            
            # SPEED OPTIMIZATION: Vectorized operations for all permutations
            random_trial_activations = activation[surrogate_bins[all_random_trial_indices]]  # Shape: (n_permutations, n_rand_trials, time_bins)
            
            # Vectorized AUC calculation for all permutations
            surro_adjusted = np.clip(random_trial_activations - thresh, a_min=0, a_max=None)
            surro_auc = np.trapz(surro_adjusted, axis=2)  # Integrate along time axis: (n_permutations, n_rand_trials)
            mean_rand_trial_auc = np.mean(surro_auc, axis=1)  # Mean across trials: (n_permutations,)
            
            all_surro_auc = surro_auc.flatten()

            # find percentile for real AUC in surrogate distribution (same logic as original)
            p_value = sum(mean_voc_auc < x for x in mean_rand_trial_auc) / len(mean_rand_trial_auc)
            p_percentile = stats.percentileofscore(mean_rand_trial_auc, mean_voc_auc)
            p_values.append(np.round(p_percentile, 1))
            s_scores.append(np.round(mean_voc_auc, 3))

            all_voc_scores.append(voc_auc)
            all_surro_scores.append(list(set(mean_rand_trial_auc)))

        # save output in a dataframe
        df = pd.DataFrame(
            {
                "concept": concept,
                "n_vocalizations": n_vocalizations,
                "activation_threshold": thresh,
                "p_value": f"({', '.join(map(str, p_values))})",
                "s_score": f"({', '.join(map(str, s_scores))})",
                "all_voc_scores": all_voc_scores,
                "all_surro_scores": all_surro_scores,
            },
            index=[0],
        )
        # Append the df for the current concept to the result_df
        result_df = pd.concat([result_df, df], ignore_index=True)
        p_stats[concept] = p_values[0]
        s_stats[concept] = s_scores[0]
    
    # save summary output
    result_df.to_csv(os.path.join(save_path, 'AUC.csv'), index=False)
    overall_p = list(p_stats.values())
    overall_s = list(s_stats.values())
    print('P: ', overall_p)
    print('S: ', overall_s)

def method_curve_shape(predictions, patient, phase, save_path, use_clusterless=False, use_lfp=False, use_combined=False, alongwith=[], predictions_length={}):
    """Generate curve shape visualization."""
    # load csv
    file = f'{save_path}/AUC.csv'
    df = pd.read_csv(file)
    df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
    p_r = list(df['p_value'])
    av_r = clean_data(list(df['all_voc_scores']))
    as_r = clean_data(list(df['all_surro_scores']))
    n_r = list(df['n_vocalizations'])
    c_r = list(df['concept'])

    plot_bins_before = 20 # how many bins to plot on either side of vocalization
    plot_bins_after = 13
    care_bins_before = 16
    care_bins_after = 1
    min_vocalizations = 2
    bin_size = 0.25
    activations = predictions
    CR_bins = []    
    if 'FR' in phase and any('CR' in element for element in alongwith):
        free_recall_windows_fr = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        free_recall_windows_cr = eval('free_recall_windows' + '_' + patient + f'_{alongwith[0]}')
        surrogate_windows_fr = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        surrogate_windows_cr = eval('surrogate_windows' + '_' + patient + f'_{alongwith[0]}')
        viewing_windows_cr = eval('viewing_windows' + '_' + patient + f'_{alongwith[0]}')
        offset = int(predictions_length[phase] * 0.25) * 1000
        viewing_bins = [np.arange(max(s, 0), max(e, 0), 0.25) * 1000 + offset for (s, e) in viewing_windows_cr]
        viewing_bins = np.concatenate(viewing_bins)
        # CR_bins = [predictions_length[phase], predictions_length[phase] + predictions_length[alongwith[0]]]
        free_recall_windows = [fr + [cr_item + offset for cr_item in cr]  for fr, cr in zip(free_recall_windows_fr, free_recall_windows_cr)]
        surrogate_windows = surrogate_windows_fr + [cr_item + offset for cr_item in surrogate_windows_cr]
    elif 'FR' in phase and not any('CR' in element for element in alongwith):
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        viewing_bins = np.array([])
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        viewing_bins = np.array([])

    temp = []
    la = free_recall_windows[0]
    ba = free_recall_windows[1]
    wh = free_recall_windows[2]
    cia = free_recall_windows[3]
    hostage = free_recall_windows[4]
    handcuff = free_recall_windows[5]
    jack = free_recall_windows[6]
    chloe = free_recall_windows[7]
    bill = free_recall_windows[8]
    fayed = free_recall_windows[9]
    amar = free_recall_windows[10]
    president = free_recall_windows[11]
    # merge Amar and Fayed
    # terrorist = fayed + amar
    # merge whiltehouse and president
    whitehouse = wh + president
    # merge CIA and Chloe
    CIA = cia + chloe
    # No LA, BombAttacks
    temp.append(whitehouse)
    temp.append(CIA)
    temp.append(hostage)
    temp.append(handcuff)
    temp.append(jack)
    temp.append(bill)
    temp.append(fayed)
    temp.append(amar)
    free_recall_windows = temp
    
    for concept_iden,vocalization_times in enumerate(free_recall_windows):
        if len(vocalization_times) <= min_vocalizations:
                # print(LABELS[concept_iden]+' did not work')
                continue

        time_bins = np.arange(0,len(activations)*bin_size, bin_size) # all the time bins

        temp_activations = []
        for i,vocal_time in enumerate(vocalization_times): # append activations around each vocalization

            # get bin closest to the vocalization time
            closest_end = np.abs(time_bins-vocal_time/1000).argmin()

            # make sure you're not at beginning or end
            if plot_bins_before < closest_end < len(time_bins) - plot_bins_after: 

                concept_acts = activations[closest_end-plot_bins_before:closest_end+plot_bins_after,concept_iden]
                temp_activations.append(concept_acts)

        sns.set_theme(style="ticks", context="paper")
        plot_bin_size = 0.25
        plot_tick_bin = 1.0
        xr = np.arange(-plot_bins_before*plot_bin_size,plot_bins_after*plot_bin_size,plot_bin_size)
        xc = np.arange(-care_bins_before*plot_bin_size,care_bins_after*plot_bin_size,plot_bin_size)

        mean_acts = np.mean(temp_activations,0)
        SE = np.std(temp_activations,0)/np.sqrt(np.shape(temp_activations)[1])
        mean_acts_care = mean_acts[plot_bins_before-care_bins_before: 21]
        SE_care = SE[plot_bins_before-care_bins_before: 21]

        time = np.arange(0, activations.shape[0], 1) * bin_size
        cr_mask = np.ones(activations.shape[0], dtype=bool)
        if viewing_bins.size == 0:
            ignore_bin = np.array([])
        else:
            ignore_bin = np.array([np.abs(time - bin_val / 1000).argmin() for bin_val in viewing_bins])
            cr_mask[ignore_bin] = False
        mean_concept_act = np.mean(activations[cr_mask, concept_iden])

        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)

        real_color = '#D66A5D'
        surro_color = '#1F466F'
        line_color = 'gray' # 'slategray'
        avg_line_color = 'firebrick'
        cmap_color = 'cividis'

        
        # curve plot
        # for significant 
        axs[0].plot(xr,mean_acts,color=line_color)
        axs[0].fill_between(xr, mean_acts-SE, mean_acts+SE, color=line_color, alpha=0.2, edgecolor='none')
        axs[0].plot(xc,mean_acts_care,color=real_color)
        axs[0].fill_between(xc, mean_acts_care-SE_care, mean_acts_care+SE_care, color=real_color, alpha=0.2, edgecolor='none')

        axs[0].plot([-plot_bins_before * plot_bin_size, plot_bins_after * plot_bin_size], 
        [mean_concept_act, mean_concept_act], linestyle='--', color=avg_line_color, alpha=0.8, label='Mean Activation')

        
        xticks = np.arange(-plot_bins_before*plot_bin_size,plot_bins_after*plot_bin_size+0.01,plot_tick_bin)
        xlabels = [int(xx) for xx in np.arange(-plot_bins_before*plot_bin_size, plot_bins_after*plot_bin_size+0.01,plot_tick_bin)]
        axs[0].set_xticks(xticks,xlabels)
        yticks = np.arange(0, 1.1, 0.2)
        axs[0].set_yticks(yticks)
        axs[0].set_ylim(0.0, 1.0)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].annotate('N = '+str(len(vocalization_times)),(plot_bins_after*plot_bin_size*0.5*0.75, 0.75))

        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Model activation')
        axs[0].legend(loc='upper right', frameon=False)


        # swarm plot
        c_name, voc, surro = c_r[concept_iden], av_r[concept_iden], as_r[concept_iden]
        mean_voc = np.mean(voc)
        std_error_voc = np.std(voc, ddof=1) / np.sqrt(len(voc))
        perc_95 = np.percentile(surro, 95)
        error_bar = axs[1].errorbar(x=['Real'], y=[mean_voc], yerr=[std_error_voc], fmt='_', color=real_color,
                markersize=12, capsize=8, elinewidth=1.5, markeredgewidth=1.5)
        show = sns.swarmplot(x=['Surrogate'] * len(surro), y=surro, ax=axs[1], color=surro_color, alpha=0.8,
                  size=1.5, dodge=False)
        show = sns.boxplot(x=['Surrogate'] * len(surro), y=surro, ax=axs[1], color=surro_color, 
                         linewidth=1.5, notch=True, width=0.7,dodge=False,showfliers=False)

        for artist in show.patches:
            col = artist.get_facecolor()
            artist.set_edgecolor(col)
            artist.set_facecolor("none")
        
            for line in show.lines[-5:]:
                line.set_color(col)
        
        axs[1].axhline(perc_95, color=avg_line_color, alpha=0.8, linestyle='--', label='95th Percentile')
        min_val, max_val = axs[1].get_ylim()
        min_val = max(min_val, 0)
        y_ticks = np.linspace(min_val, max_val, 6)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels([f"{y:.1f}" for y in y_ticks])
        axs[1].set_ylabel('Area')
        axs[1].legend(loc='upper right', frameon=False)

        sns.despine()
        plt.tight_layout()

        label_without_punctuation = re.sub(r'[^\w\s]','',LABELS8[concept_iden])

        if LABELS8[concept_iden] in ['CIA']:
            print()
            np.savez(f'paper_figures_making/figure2_materials/{patient}_{phase}_{label_without_punctuation}.npz',
                     free_recall_window=np.array(free_recall_windows[concept_iden]), activation=activations[:, concept_iden], viewing_bins=viewing_bins,
                     c_r=c_r[concept_iden], av_r=np.array(av_r[concept_iden]), as_r=np.array(as_r[concept_iden]))

        fig.savefig(os.path.join(save_path, f'{label_without_punctuation}.png'),
            bbox_inches='tight', dpi=200)
        plt.cla()
        plt.clf()   
        plt.close()

def method_curve_shape_all(predictions, patient, phase, save_path, use_clusterless=False, use_lfp=False, use_combined=False, alongwith=[], predictions_length={}):
    """Generate all curve shape visualizations."""
    plot_bins_before = 20 # how many bins to plot on either side of vocalization
    plot_bins_after = 13
    care_bins_before = 16
    care_bins_after = 1
    min_vocalizations = 2
    bin_size = 0.25
    activations = predictions
    CR_bins = []    
    if 'FR' in phase and any('CR' in element for element in alongwith):
        free_recall_windows_fr = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        free_recall_windows_cr = eval('free_recall_windows' + '_' + patient + f'_{alongwith[0]}')
        surrogate_windows_fr = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        surrogate_windows_cr = eval('surrogate_windows' + '_' + patient + f'_{alongwith[0]}')
        viewing_windows_cr = eval('viewing_windows' + '_' + patient + f'_{alongwith[0]}')
        offset = int(predictions_length[phase] * 0.25) * 1000
        viewing_bins = [np.arange(max(s, 0), max(e, 0), 0.25) * 1000 + offset for (s, e) in viewing_windows_cr]
        viewing_bins = np.concatenate(viewing_bins)
        # CR_bins = [predictions_length[phase], predictions_length[phase] + predictions_length[alongwith[0]]]
        free_recall_windows = [fr + [cr_item + offset for cr_item in cr]  for fr, cr in zip(free_recall_windows_fr, free_recall_windows_cr)]
        surrogate_windows = surrogate_windows_fr + [cr_item + offset for cr_item in surrogate_windows_cr]
    elif 'FR' in phase and not any('CR' in element for element in alongwith):
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        viewing_bins = np.array([])
    else:
        free_recall_windows = eval('free_recall_windows' + '_' + patient + f'_{phase}')
        surrogate_windows = eval('surrogate_windows' + '_' + patient + f'_{phase}')
        viewing_bins = np.array([])

    temp = []
    la = free_recall_windows[0]
    ba = free_recall_windows[1]
    wh = free_recall_windows[2]
    cia = free_recall_windows[3]
    hostage = free_recall_windows[4]
    handcuff = free_recall_windows[5]
    jack = free_recall_windows[6]
    chloe = free_recall_windows[7]
    bill = free_recall_windows[8]
    fayed = free_recall_windows[9]
    amar = free_recall_windows[10]
    president = free_recall_windows[11]
    # merge Amar and Fayed
    # terrorist = fayed + amar
    # merge whiltehouse and president
    whitehouse = wh + president
    # merge CIA and Chloe
    CIA = cia + chloe
    # No LA, BombAttacks
    temp.append(whitehouse)
    temp.append(CIA)
    temp.append(hostage)
    temp.append(handcuff)
    temp.append(jack)
    temp.append(bill)
    temp.append(fayed)
    temp.append(amar)
    free_recall_windows = temp
    
    fig, axs = plt.subplots(3, 3, figsize=(9,9), sharey=True)
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    real_color = '#E86156'
    surro_color = '#1F466F'
    line_color = 'gray' # 'slategray'
    avg_line_color = 'firebrick'
    cmap_color = 'cividis'
    
    for concept_iden,vocalization_times in enumerate(free_recall_windows):
        ax = axs.flat[concept_iden]
        if len(vocalization_times) <= min_vocalizations:
                ax.set_title(LABELS8[concept_iden])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                continue

        time_bins = np.arange(0,len(activations)*bin_size, bin_size) # all the time bins

        temp_activations = []
        for i,vocal_time in enumerate(vocalization_times): # append activations around each vocalization

            # get bin closest to the vocalization time
            closest_end = np.abs(time_bins-vocal_time/1000).argmin()

            # make sure you're not at beginning or end
            if plot_bins_before < closest_end < len(time_bins) - plot_bins_after: 

                concept_acts = activations[closest_end-plot_bins_before:closest_end+plot_bins_after,concept_iden]
                temp_activations.append(concept_acts)

        plot_bin_size = 0.25
        plot_tick_bin = 1.0
        xr = np.arange(-plot_bins_before*plot_bin_size,plot_bins_after*plot_bin_size,plot_bin_size)
        xc = np.arange(-care_bins_before*plot_bin_size,care_bins_after*plot_bin_size,plot_bin_size)

        mean_acts = np.mean(temp_activations,0)
        SE = np.std(temp_activations,0)/np.sqrt(np.shape(temp_activations)[1])
        mean_acts_care = mean_acts[plot_bins_before-care_bins_before: 21]
        SE_care = SE[plot_bins_before-care_bins_before: 21]

        time = np.arange(0, activations.shape[0], 1) * bin_size
        cr_mask = np.ones(activations.shape[0], dtype=bool)
        if viewing_bins.size == 0:
            ignore_bin = np.array([])
        else:
            ignore_bin = np.array([np.abs(time - bin_val / 1000).argmin() for bin_val in viewing_bins])
            cr_mask[ignore_bin] = False
        mean_concept_act = np.mean(activations[cr_mask, concept_iden])
      
        # for significant 
        ax.plot(xr,mean_acts,color=line_color)
        ax.fill_between(xr, mean_acts-SE, mean_acts+SE, color=line_color, alpha=0.2, edgecolor='none', label='_nolegend_')
        ax.plot(xc,mean_acts_care,color=real_color)
        ax.fill_between(xc, mean_acts_care-SE_care, mean_acts_care+SE_care, color=real_color, alpha=0.2, edgecolor='none', label='_nolegend_')

        ax.plot([-plot_bins_before * plot_bin_size, plot_bins_after * plot_bin_size], 
        [mean_concept_act, mean_concept_act], linestyle='--', color=avg_line_color, alpha=0.8, label='Mean Activation')
        
        xticks = np.arange(-plot_bins_before*plot_bin_size,plot_bins_after*plot_bin_size+0.01,plot_tick_bin)
        xlabels = [int(xx) for xx in np.arange(-plot_bins_before*plot_bin_size, plot_bins_after*plot_bin_size+0.01,plot_tick_bin)]
        ax.set_xticks(xticks,xlabels)
        yticks = np.arange(0, 1.1, 0.2)
        ax.set_yticks(yticks)
        ax.set_ylim(0.0, 1.0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.annotate('N = '+str(len(vocalization_times)),(plot_bins_after*plot_bin_size*0.5*0.75, 0.75))

        ax.set_title(LABELS8[concept_iden])
        ax.set_xlabel('Time (s)')

    sns.despine()
    plt.tight_layout()

    fig.savefig(os.path.join(save_path, 'all_concepts.png'),
        bbox_inches='tight', dpi=200)
    plt.cla()
    plt.clf()   
    plt.close()
