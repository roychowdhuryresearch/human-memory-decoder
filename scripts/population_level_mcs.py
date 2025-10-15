import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_versions = 'simulated'
model_version = 'CSA6_CCA6_384_H6'
epoch = '49'
lesion = 'Full' 
                              
def load_participant_data(participant, data_version): 
    """Load data for a single participant and model version"""
    p_r1, p_r2 = [np.nan] * 8, [np.nan] * 8
    n_r1, n_r2 = [0] * 8, [0] * 8 
    
    # Use cr='0' for participants p1 and p2, cr='1' for others
    participant_cr = '0' if participant in ['p1', 'p2'] else '1'
    
    # FR1
    file = f'results/{model_version}/{participant}_clusterless_transformer_{lesion}/memory/{data_version}/epoch{epoch}_FR1_{participant_cr}/AUC.csv'
    if os.path.exists(file):
        df = pd.read_csv(file)
        df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
        p_r1 = list(df['p_value'])
        n_r1 = list(df['n_vocalizations'])
    
    # FR2
    file = f'results/{model_version}/{participant}_clusterless_transformer_{lesion}/memory/{data_version}/epoch{epoch}_FR2_{participant_cr}/AUC.csv'
    if os.path.exists(file):
        df = pd.read_csv(file)
        df['p_value'] = df['p_value'].str.strip('()').apply(pd.to_numeric, errors='coerce')
        p_r2 = list(df['p_value'])
        n_r2 = list(df['n_vocalizations'])
    
    return p_r1, p_r2, n_r1, n_r2

# List of participants
participants = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
data_dict = {p: load_participant_data(p, data_versions) for p in participants}

def create_matrices_from_manual_config():
    num_concepts = 8
    num_participants = len(participants)
    
    # Initialize matrices
    p_matrix_r1 = np.zeros((num_concepts, num_participants))
    p_matrix_r2 = np.zeros((num_concepts, num_participants))
    n_matrix_r1 = np.zeros((num_concepts, num_participants))
    n_matrix_r2 = np.zeros((num_concepts, num_participants))
    
    # For each participant
    for p_idx, participant in enumerate(participants):
        n_r1 = data_dict[participant][2]
        n_r2 = data_dict[participant][3]
                
        # Set FR1 scores
        p_matrix_r1[:, p_idx] = data_dict[participant][0]
        n_r1 = data_dict[participant][2]
            
        # Set FR2 scores
        p_matrix_r2[:, p_idx] = data_dict[participant][1]
        n_r2 = data_dict[participant][3]
        
        n_matrix_r1[:, p_idx] = n_r1
        n_matrix_r2[:, p_idx] = n_r2
        
    return p_matrix_r1, p_matrix_r2, n_matrix_r1, n_matrix_r2

p_matrix_r1, p_matrix_r2, n_matrix_r1, n_matrix_r2 = create_matrices_from_manual_config()

participants = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
classes = ['WhiteHouse', 'CIA', 'Sacrifice', 'Handcuff', 'J.Bauer', 'B.Buchanan', 'A.Fayed', 'A.Amar']

unified_fontsize = 11
colors_fancy = {
    "cinnabar": "#E34234",
    "dark_pastel_blue": "#779ECB",
    "moonstone_blue": "#73A9C2",
    "brass": "#B5A642",
    "real": '#D66A5D',
    "surro": '#1F466F',
    "pre_sleep": "#626B3E",
    "post_sleep": "#B2AF7A",
    "avg_line": 'black',
    "curve_line":'gray',
}
swarm_color1, swarm_color2 = colors_fancy['pre_sleep'], colors_fancy['post_sleep']

def wilcoxon_z_value(data, zero_method='wilcox'):
    """Compute Wilcoxon signed-rank statistic (W), p-value, and z using normal approximation."""
    W, p = stats.wilcoxon(data, zero_method=zero_method, mode='auto')  
    
    if zero_method == 'wilcox':
        data = data[data != 0]
    elif zero_method == 'pratt':
        pass
    
    n = len(data)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n*(n+1)*(2*n+1) / 24.0)
    z = (W - mu) / sigma
    return W, p, z

def plot_pval_grid(p_vals, n_voc, save_fig_flag=False, svg_suffix="FR1", ax=None, fig=None, heatmap_label=False):
    n_cons = p_vals.shape[0]
    n_sub = p_vals.shape[1]
    sig_thresh = 95

    mask = np.isnan(p_vals)
    heatmap_line_color = swarm_color1 if svg_suffix == 'Pre-sleep recall' else swarm_color2
    im = sns.heatmap(
        p_vals,
        mask=mask,
        vmin=0,
        vmax=100,
        cmap='cividis',
        square=False,
        cbar=False,
        ax=ax,
        linewidths=0.5,
        linecolor=heatmap_line_color,
        xticklabels=participants,
        yticklabels=classes,
    )

    norm = Normalize(vmin=0, vmax=100)
    cmap = colormaps['cividis']
    for i in range(n_cons):
        for j in range(n_sub):
            val = p_vals[i, j]
            rgba = cmap(norm(val))
            brightness = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
            text_color = 'black' if brightness > 0.5 else 'white'
            if np.isnan(val):
                text_color = 'black'
            ax.text(j + 0.5, i + 0.5, f"{n_voc[i, j].astype(int)}", color=text_color, fontsize=unified_fontsize, ha='center', va='center')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=unified_fontsize)
    if heatmap_label:
        cbar.set_label('Memory confidence score', fontsize=unified_fontsize)

    ax.set_xlabel('Participant', fontsize=unified_fontsize)
    ax.set_ylabel("Concept", fontsize=unified_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=unified_fontsize)
    ax.grid(False)

def plot_swarm(p_vals, n_voc, ax=None, fig=None, color='black'):
    flattened_data = p_vals.flatten()
    flattened_number = n_voc.flatten()
    mask = flattened_number > 2
    valid_data = flattened_data[mask]
    
    one_sample_t_stat, p_value = stats.ttest_1samp(valid_data, 50)
    _, one_sample_wilcoxon_p, z = wilcoxon_z_value(valid_data - 50)
    cohen_d = (valid_data.mean() - 50) / valid_data.std(ddof=1)

    print(f"t-statistic: {one_sample_t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"p-value (wilcoxon): \033[1m{one_sample_wilcoxon_p:.4f}\033[0m")
    print(f"wilcoxon: Z = {-z:.3f}")
    print(f"Cohen's d: {cohen_d:.3f}")

    sns.swarmplot(x=[0]*len(valid_data), y=valid_data, ax=ax, color=color, size=3.3, alpha=1.0, zorder=1)

    mean_value = np.mean(valid_data)
    std_dev = np.std(valid_data)
    n = len(valid_data)
    ci_value = 1.96 * (std_dev / np.sqrt(n))
    median_value = np.median(valid_data)
    iqr_lower = np.percentile(valid_data, 25)
    iqr_upper = np.percentile(valid_data, 75)
 
    print(median_value)

    ax.plot([-0.25, 0.25], [median_value, median_value], color='black', lw=2, linestyle='--', label="Median", zorder=2)
    ax.fill_between([-0.25, 0.25], iqr_lower, iqr_upper, color=color, alpha=0.6, label="IQR", zorder=1)

    ax.text(0, max(valid_data) * 1.06, f'p-value: {one_sample_wilcoxon_p:.4f}', fontsize=unified_fontsize-1, ha='center', color='black')
    ax.text(0, max(valid_data) * 1.12, f'Cohen\'s d: {cohen_d:.4f}', fontsize=unified_fontsize-1, ha='center', color='black')
    ax.text(0, max(valid_data) * 1.18, f'median: {median_value:.4f}', fontsize=unified_fontsize-1, ha='center', color='black')

    y_ticks = np.linspace(0, 100, 3)
    ax.set_yticks(y_ticks)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.6, color='lightgray')
    ax.tick_params(axis='y', labelsize=unified_fontsize)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-2, 102)
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

if __name__ == "__main__":
    # Required imports for plotting
    import seaborn as sns
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.colors import Normalize
    from matplotlib import colormaps
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy import stats

    fig = plt.figure(figsize=(15, 4))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 8
    plt.rcParams["svg.fonttype"] = "none"

    gs = GridSpec(1, 2, width_ratios=[1, 1])

    gs_ax1 = GridSpecFromSubplotSpec(1, 2, width_ratios=[8, 1], subplot_spec=gs[0, 0], wspace=0.3)
    ax1a = fig.add_subplot(gs_ax1[0, 0])
    ax1b = fig.add_subplot(gs_ax1[0, 1])

    gs_ax2 = GridSpecFromSubplotSpec(1, 2, width_ratios=[8, 1], subplot_spec=gs[0, 1], wspace=0.3)
    ax2a = fig.add_subplot(gs_ax2[0, 0])
    ax2b = fig.add_subplot(gs_ax2[0, 1])

    plot_pval_grid(p_matrix_r1, n_matrix_r1, save_fig_flag=False, svg_suffix="Pre-sleep recall", ax=ax1a, fig=fig, heatmap_label=False)
    plot_pval_grid(p_matrix_r2, n_matrix_r2, save_fig_flag=False, svg_suffix="Post-sleep recall", ax=ax2a, fig=fig, heatmap_label=False)
    plot_swarm(p_matrix_r1, n_matrix_r1, ax=ax1b, fig=fig, color=swarm_color1)
    plot_swarm(p_matrix_r2, n_matrix_r2, ax=ax2b, fig=fig, color=swarm_color2)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/figure2d.png", format="png", bbox_inches="tight") 


    metadata = {
        "Concept":[],
        "Patient":[],
        "Confidence":[],
        "Number":[],
        "Model Type":[],
        "Recall Phase":[],
    }
    for pi, p in enumerate(participants):
        confs = p_matrix_r1[:, pi]
        times = n_matrix_r1[:, pi]
        for ci, c in enumerate(confs):
            metadata['Concept'].append(classes[ci])
            metadata['Patient'].append(p)
            metadata['Confidence'].append(c)
            metadata['Number'].append(times[ci])
            metadata['Model Type'].append('Full')
            metadata['Recall Phase'].append('R1')

    for pi, p in enumerate(participants):
        confs = p_matrix_r2[:, pi]
        times = n_matrix_r2[:, pi]
        for ci, c in enumerate(confs):
            metadata['Concept'].append(classes[ci])
            metadata['Patient'].append(p)
            metadata['Confidence'].append(c)
            metadata['Number'].append(times[ci])
            metadata['Model Type'].append('Full')
            metadata['Recall Phase'].append('R2')

    metadata_df = pd.DataFrame(metadata)
    
    # # Check if file exists and read existing data
    filename = "metadata_data.xlsx"
    # if os.path.exists(filename):
    #     existing_df = pd.read_excel(filename)
    #     # Concatenate existing data with new data
    #     metadata_df = pd.concat([existing_df, metadata_df], ignore_index=True)
    
    # # Save the combined dataframe
    metadata_df.to_excel(filename, index=False)