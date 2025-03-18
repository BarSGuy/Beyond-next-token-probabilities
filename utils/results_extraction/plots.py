import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

### --- BookMIA Results --- ###
def plot_heatmaps_for_bookmia():
    bookmia_zero_shot_data = np.array([
        # P-6.9b # P-12b # L-13b # L-30b
        [90.71, 87.36, 78.70, 77.28],
        [89.86, 89.43, 83.36, 85.06],
        [78.31, 78.26, 91.03, 95.60],
        [86.27, 83.61, 90.48, 95.60]
    ])

    bookmia_zero_shot_error = np.array([
        [0.9, 1.41, 1.44, 1.94],
        [0.21, 0.59, 1.38, 3.49],
        [2.22, 1.38, 0.15, 0.33],
        [0.63, 1.53, 0.56, 0.41]
    ])
    bookmia_better_than_from_scratch_indicators = np.array([
        ['*', '', '', ''],
        ['*', '', '', ''],
        ['', '', '*', '*'],
        ['', '', '*', '*']
    ])
    bookmia_better_than_baselines_indicators = np.array([
        ['*', '*', '*', ''],
        ['*', '*', '*', ''],
        ['*', '*', '*', '*'],
        ['*', '*', '*', '*']
    ])
    plot_heatmap(data=bookmia_zero_shot_data, errors=bookmia_zero_shot_error, better_than_from_scratch_indicators=bookmia_better_than_from_scratch_indicators, better_than_baselines_indicators=bookmia_better_than_baselines_indicators,
                 labels = ['P-6.9b', 'P-12b', 'L-13b', 'L-30b'],
                 file_name="./Results/heatmaps/BookMIA/bookmia_zero_shot_heatmap")

### -- Cross-Dataset Results -- ###
# Meta - finetune
def plot_heatmaps_for_meta_fine_tune():
    meta_fine_tune_data = np.array([
        # imdb # movies # hotpotqa
        [89.44, 62.96, 56.52],
        [86.09, 77.04, 68.40],
        [85.54, 75.21, 72.97]
    ])

    meta_fine_tune_errors = np.array([
        [0.32, 0.32, 0.88],
        [0.50, 0.77, 0.47],
        [1.32, 0.18, 0.41]
    ])

    meta_better_than_baselines_indicators = np.array([
        ['*', '', ''],
        ['*', '*', '*'],
        ['*', '*', '*']
    ])
    meta_better_than_from_scratch_indicators = np.array([
        ['*', '*', ''],
        ['*', '*', '*'],
        ['*', '*', '*']
    ])
    plot_heatmap(data=meta_fine_tune_data, errors=meta_fine_tune_errors, better_than_from_scratch_indicators=meta_better_than_from_scratch_indicators, better_than_baselines_indicators=meta_better_than_baselines_indicators,
                 labels = ['IMDB', 'Movies', 'HotpotQA'],
                 file_name="./Results/heatmaps/Cross_datasets/meta_fine_tune_heatmap",
                 title="L-3-8b")
    
# Meta - from scratch
def plot_heatmaps_for_meta_from_scratch():
    meta_from_scratch_data = np.array([
        [76.53, 61.26, 57.79],
        [76.23, 61.54, 60.62],
        [72.63, 60.83, 57.73]
    ])

    meta_from_scratch_errors = np.array([
        [4.37, 0.70, 1.30],
        [6.56, 0.66, 1.22],
        [6.35, 0.91, 1.76]
    ])

    plot_heatmap(data=meta_from_scratch_data, errors=meta_from_scratch_errors,
                 labels = ['IMDB', 'Movies', 'HotpotQA'],
                 file_name="./Results/heatmaps/Cross_datasets/meta_scratch_heatmap",
                 title="L-3-8b")

# Mistral - finetune
def plot_heatmaps_for_mistral_fine_tune():
    mistral_fine_tune_data = np.array([
        # imdb # movies # hotpotqa
        [96.11, 63.60, 68.37],
        [89.97, 68.59, 67.37],
        [93.78, 71.19, 73.24]
    ])

    mistral_fine_tune_errors = np.array([
        [0.03, 0.51, 0.65],
        [0.26, 1.08, 0.17],
        [0.17, 0.14, 0.28]
    ])

    mistral_better_than_baselines_indicators = np.array([
            ['*', '', '*'],
            ['*', '*', '*'],
            ['*', '*', '*']
        ])
    mistral_better_than_from_scratch_indicators = np.array([
            ['*', '*', '*'],
            ['*', '*', '*'],
            ['*', '*', '*']
        ])
    plot_heatmap(data=mistral_fine_tune_data, errors=mistral_fine_tune_errors, better_than_from_scratch_indicators=mistral_better_than_from_scratch_indicators, better_than_baselines_indicators=mistral_better_than_baselines_indicators,
                 labels = ['IMDB', 'Movies', 'HotpotQA'],
                 file_name="./Results/heatmaps/Cross_datasets/mistral_fine_tune_heatmap",
                 title="Mis-7b")

# Mistral - from scratch
def plot_heatmaps_for_mistral_from_scratch():
    mistral_from_scratch_data = np.array([
        [90.45, 59.21, 63.19],
        [87.88, 59.51, 63.31],
        [87.97, 60.04, 63.29]
    ])

    mistral_from_scratch_errors = np.array([
        [0.40, 1.83, 0.46],
        [1.58, 1.79, 0.77],
        [1.39, 1.64, 0.73]
    ])
    plot_heatmap(data=mistral_from_scratch_data, errors=mistral_from_scratch_errors,
                 labels = ['IMDB', 'Movies', 'HotpotQA'],
                 file_name="./Results/heatmaps/Cross_datasets/mistral_scratch_heatmap",
                 title="Mis-7b")

###################################

### -- Cross-Models -- ###
# IMDB - fine-tune
def plot_heatmaps_for_imdb_fine_tune():
    imdb_fine_tune_data = np.array([
        # meta # mistral
        [89.44, 91.12],
        [84.91, 96.11]
    ])
    imdb_from_fine_tune_errors = np.array([
        [0.32, 0.32],
        [0.46, 0.03]
    ])
    imdb_better_than_baselines_indicators = np.array([
        ['*', '*'],
        ['*', '*']
    ])
    imdb_better_than_from_scratch_indicators = np.array([
        ['*', '*'],
        ['*', '*']

    ])
    plot_heatmap(data=imdb_fine_tune_data, errors=imdb_from_fine_tune_errors,
                better_than_from_scratch_indicators=imdb_better_than_from_scratch_indicators, better_than_baselines_indicators=imdb_better_than_baselines_indicators,
                labels = ['L-3-8b', 'Mis-7b'],
                 file_name="./Results/heatmaps/Cross_models/imdb_fine_tune_heatmap",
                 title='IMDB')

# IMDB - from scratch
def plot_heatmaps_for_imdb_from_scratch():
    imdb_from_scratch_data = np.array([
        # meta # mistral
        [76.53, 86.37],
        [77.26, 90.45]
    ])
    imdb_from_scratch_errors = np.array([
        [4.37, 5.81],
        [3.56, 0.40]
    ])
    plot_heatmap(data=imdb_from_scratch_data, errors=imdb_from_scratch_errors,
                 labels = ['L-3-8b', 'Mis-7b'],
                 file_name="./Results/heatmaps/Cross_models/imdb_scratch_heatmap",
                 title='IMDB')

# Movies - fine-tune
def plot_heatmaps_for_movies_fine_tune():
    movies_fine_tune_data = np.array([
        # meta # mistral
        [77.04, 66.28],
        [69.77, 68.59]
    ])
    movies_from_fine_tune_errors = np.array([
        [0.77, 0.26],
        [0.29, 1.08]
    ])
    movies_better_than_baselines_indicators = np.array([
        ['*', '*'],
        ['', '*']
    ])
    movies_better_than_from_scratch_indicators = np.array([
        ['*', '*'],
        ['*', '*']

    ])
    plot_heatmap(data=movies_fine_tune_data, errors=movies_from_fine_tune_errors,
                better_than_from_scratch_indicators=movies_better_than_from_scratch_indicators, better_than_baselines_indicators=movies_better_than_baselines_indicators,
                labels = ['L-3-8b', 'Mis-7b'],
                 file_name="./Results/heatmaps/Cross_models/movies_fine_tune_heatmap",
                 title='Movies')
    
# Movies - from scratch
def plot_heatmaps_for_movies_from_scratch():
    movies_from_scratch_data = np.array([
        # meta # mistral
        [61.54, 59.23],
        [60.21, 59.51]
    ])
    movies_from_scratch_errors = np.array([
        [0.66, 1.10],
        [1.28, 1.79]
    ])
    plot_heatmap(data=movies_from_scratch_data, errors=movies_from_scratch_errors,
                 labels = ['L-3-8b', 'Mis-7b'],
                 file_name="./Results/heatmaps/Cross_models/movies_scratch_heatmap",
                 title='Movies')

# HotpotQA - fine-tune
def plot_heatmaps_for_hotpotqa_fine_tune():
    hotpotqa_fine_tune_data = np.array([
        # meta # mistral
        [72.97, 68.42],
        [64.41, 73.24]
    ])
    hotpotqa_from_fine_tune_errors = np.array([
        [0.41, 1.10],
        [0.54, 0.28]
    ])
    hotpotqa_better_than_baselines_indicators = np.array([
        ['*', '*'],
        ['', '*']
    ])
    hotpotqa_better_than_from_scratch_indicators = np.array([
        ['*', '*'],
        ['*', '*']

    ])
    plot_heatmap(data=hotpotqa_fine_tune_data, errors=hotpotqa_from_fine_tune_errors,
                better_than_from_scratch_indicators=hotpotqa_better_than_from_scratch_indicators, better_than_baselines_indicators=hotpotqa_better_than_baselines_indicators,
                labels = ['L-3-8b', 'Mis-7b'],
                 file_name="./Results/heatmaps/Cross_models/hotpotqa_fine_tune_heatmap",
                 title='HotpotQA')

def plot_heatmaps_for_hotpotqa_from_scratch():
    hotpotqa_from_scratch_data = np.array([
        # meta # mistral
        [57.73, 63.34],
        [57.10, 63.29]
    ])
    hotpotqa_from_scratch_errors = np.array([
        [1.76, 0.27],
        [2.05, 0.73]
    ])
    plot_heatmap(data=hotpotqa_from_scratch_data, errors=hotpotqa_from_scratch_errors,
                 labels = ['L-3-8b', 'Mis-7b'],
                 file_name="./Results/heatmaps/Cross_models/hotpotqa_scratch_heatmap",
                 title='HotpotQA')
##########################

### -- Ablation -- ###
def ablation_study_topk(file_name="./Results/heatmaps/Ablation_study/Topk_Ablation_Study"):
    Colors = {
        "mistral": 'lightblue',
        'meta': 'green',
        'pythia': 'grey',
        'LLaMa': 'purple',
    }

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    data = {
        "topk": ["10", "20", "50", "100", "300", "500", "800", "1000", "Full Vocabulary"],
        "imdb_mistral": [95.22, 95.85, 95.77, 95.86, 95.42, 95.69, 95.87, 96.11, 95.97],
        "imdb_meta": [89.09, 89.73, 90.15, 89.70, 88.65, 89.58, 89.74, 89.44, 88.75],
        "hotpotqa_mistral": [73.16, 71.29, 72.54, 72.42, 72.55, 71.32, 72.97, 73.24, 72.83],
        "hotpotqa_meta": [71.25, 71.94, 69.56, 71.58, 71.05, 72.76, 71.98, 72.97, 74.28],
    }

    # Example std values for k=1000
    std_values = {
        "imdb_mistral": 0.03,
        "imdb_meta": 0.28,
        "hotpotqa_mistral": 0.32,
        "hotpotqa_meta": 0.41,
    }

    def add_min_max_lines_and_gap(ax, values, topk, color):
        min_val = min(values)
        max_val = max(values)
        min_topk = topk[values.index(min_val)]
        max_topk = topk[values.index(max_val)]
        gap = max_val - min_val

        ax.axhline(y=min_val, color='black', linestyle='--', label=f"Min Bin ({min_topk})")
        ax.axhline(y=max_val, color='gray', linestyle='--', label=f"Max Bin ({max_topk})")
        ax.text(1.02, 1.1, f"(Gap: {gap:.2f})", color=color, fontsize=16, ha='right', va='top', transform=ax.transAxes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams.update({'font.size': 16})

    plot_info = [
        (axes[0, 0], "hotpotqa_meta", "HotpotQA: L-3-8b", Colors['meta']),
        (axes[0, 1], "imdb_meta", "IMDB: L-3-8b", Colors['meta']),
        (axes[1, 0], "hotpotqa_mistral", "HotpotQA: Mis-7b", Colors['mistral']),
        (axes[1, 1], "imdb_mistral", "IMDB: Mis-7b", Colors['mistral']),
    ]

    for ax, key, title, color in plot_info:
        bars = ax.bar(data["topk"], data[key], color=color, alpha=0.7)
        add_min_max_lines_and_gap(ax, data[key], data["topk"], color='black')
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel("AUC", fontsize=18)
        ax.set_xticklabels(data["topk"], rotation=45, fontsize=14)
        ax.legend(fontsize=14)

        # Add std error bar only at k=1000
        idx_1000 = data["topk"].index("1000")
        ax.errorbar("1000", data[key][idx_1000], yerr=std_values[key], fmt='none', ecolor='red', capsize=5, linewidth=2)

    for i in range(2):
        for j in range(2):
            axes[i, j].set_ylim(65, None)

    plt.tight_layout()
    plt.savefig(file_name + ".pdf", dpi=300, bbox_inches='tight')
    plt.show()


def ablation_on_three_arch(file_name='./Results/heatmaps/Ablation_study/Ablation_baselines'):
    # Example color dictionary
    Colors = {
        "mistral": 'lightblue',
        'meta': 'green',
        'pythia': 'grey',
        'LLaMa': 'purple',
    }
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # Data including standard deviation for histograms
    data = {
        "legends": ["LOS-Net", "ATP-R-Tranf.", "ATP-R-MLP"],
        "mistral": { 
            "hotpotqa": ([73.24, 63.80, 61.36], [0.29, 0.98, 0.33]),
            "imdb": ([96.11, 92.30, 88.95], [0.03, 1.66, 0.40]),
            "movies": ([68.69, 62.41, 60.63], [1.09, 0.22, 0.16]),
        },
        "meta": {
            "hotpotqa": ([72.97, 61.39, 60.09], [0.41, 1.24, 0.24]),
            "imdb": ([89.44, 82.56, 85.28], [0.32, 0.63, 0.49]),
            "movies": ([77.04, 64.95, 67.19], [0.77, 0.68, 0.25]),
        },
        "Pythia-6.9b": {
            "BookMIA": ([90.71, 79.59, 56.31], [0.91, 0.61, 1.48]),
        },
        "Pythia-12b": {
            "BookMIA": ([89.43, 74.77, 57.19], [0.59, 0.57, 1.06]),
        },
        "LLaMa-13b": {
            "BookMIA": ([91.03, 74.65, 66.60], [0.16, 0.79, 1.05]),
        },
        "LLaMa-30b": {
            "BookMIA": ([95.60, 87.62, 83.89], [0.40, 0.68, 0.41]),
        }
    }
        
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Modified plotting function to include error bars
    def plot_with_error(ax, values, errors, labels, title, color):
        ax.bar(labels, values, yerr=errors, color=color, alpha=0.7, capsize=5)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel("AUC", fontsize=18)
        ax.set_xticklabels(labels, rotation=45, fontsize=14)

    # Example usage of the new plotting function
    plot_with_error(axes[0, 0], *data["mistral"]["hotpotqa"], data["legends"], "HotpotQA: Mis-7b", Colors['mistral'])
    plot_with_error(axes[0, 1], *data["mistral"]["imdb"], data["legends"], "IMDB: Mis-7b", Colors['mistral'])
    plot_with_error(axes[0, 2], *data["mistral"]["movies"], data["legends"], "Movies: Mis-7b", Colors['mistral'])

    plot_with_error(axes[1, 0], *data["meta"]["hotpotqa"], data["legends"], "HotpotQA: L-3-8b", Colors['meta'])
    plot_with_error(axes[1, 1], *data["meta"]["imdb"], data["legends"], "IMDB: L-3-8b", Colors['meta'])
    plot_with_error(axes[1, 2], *data["meta"]["movies"], data["legends"], "Movies: L-3-8b", Colors['meta'])

    plot_with_error(axes[0, 3], *data["Pythia-6.9b"]["BookMIA"], data["legends"], "BookMIA: Pythia-6.9b", Colors['pythia'])
    plot_with_error(axes[1, 3], *data["Pythia-12b"]["BookMIA"], data["legends"], "BookMIA: Pythia-12b", Colors['pythia'])
    plot_with_error(axes[0, 4], *data["LLaMa-13b"]["BookMIA"], data["legends"], "BookMIA: LLaMa-13b", Colors['LLaMa'])
    plot_with_error(axes[1, 4], *data["LLaMa-30b"]["BookMIA"], data["legends"], "BookMIA: LLaMa-30b", Colors['LLaMa'])

    # Adjust y-axis limits
    for ax in axes.flatten():
        ax.set_ylim(55, None)

    plt.tight_layout()
    plt.savefig(file_name + ".pdf", dpi=300, bbox_inches='tight')
    plt.show()

##########################
def plot_heatmap(
    data = np.array([
        [0, 0.6126, 0.5779],
        [0.7623, 0, 0.6062],
        [0.7263, 0.6083, 0]
    ]),
    errors = np.array([
        [0, 0.0015, 0.0025],
        [0.0080, 0, 0.0020],
        [0.0075, 0.0022, 0]
    ]),
    better_than_from_scratch_indicators = np.array([
        ['', '', ''],
        ['', '', ''],
        ['', '', '']
    ]),
    better_than_baselines_indicators = np.array([
        ['', '', ''],
        ['', '', ''],
        ['', '', '']
    ]),
    labels = ['IMDB', 'Movies', 'HotpotQA'],
    file_name="heatmap",
    title=""
):
    df = pd.DataFrame(data, index=labels, columns=labels)

    # Create a mask for diagonal elements
    if all(data[i][i] == 0 for i in range(len(data))):
        mask = np.eye(len(labels), dtype=bool)
    else:
        mask = None
        

    plt.figure(figsize=(8, 6))
    # cmap = "RdBu"
    cmap = "coolwarm_r"
    # cmap = "RdBu_r"
    # cmap = "bwr_r"
    
    # Plot heatmap
    ax = sns.heatmap(df, annot=False, fmt='', cmap=plt.cm.RdBu, linewidths=0.0, cbar=True, mask=mask)
    if title != '':
        plt.title(title, fontsize=40)
    # Get colormap
    cmap = ax.collections[0].get_cmap()
    norm = ax.collections[0].norm  # Normalize values

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = data[i, j]
            error = errors[i, j]
            single_star = better_than_from_scratch_indicators[i, j]
            bold = better_than_baselines_indicators[i, j] == '*'

            if i == j and value == 0:  # Diagonal values (masked)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='#CCCCCC', hatch='//', edgecolor='black', lw=0))
                # ax.text(j + 0.5, i + 0.5, "N/A", ha='center', va='center', fontsize=25, color="black")
                ax.plot([j + 0.25, j + 0.75], [i + 0.5, i + 0.5], linestyle="--", color="black", linewidth=2.5)
            else:
                formatted_value = f'{value:.2f}'
                if bold:
                    formatted_value = r"$\bf{" + formatted_value + r"}$"

                text = f'{formatted_value} {single_star}'.strip()
                err_text = f'± {error:.2g}'

                # Get background color from heatmap
                bg_color = cmap(norm(value))  # Get RGBA color
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]  # Perceived brightness

                text_color = "white" if luminance < 0.5 else "black"  # Choose text color based on luminance

                if len(labels) == 2:
                    fontsize = 30
                    error_fontsize = 20
                elif len(labels) == 3:
                    fontsize = 26
                    error_fontsize = 20
                elif len(labels) == 4:
                    fontsize = 18
                    error_fontsize = 12
                ax.text(j + 0.5, i + 0.45, text, ha='center', va='center', fontsize=fontsize, color=text_color)
                ax.text(j + 0.5, i + 0.65, err_text, ha='center', va='center', fontsize=error_fontsize, color=text_color)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)  # Change label size here
    ax.tick_params(axis='x', labelsize=20)  # size of x-axis tick labels
    ax.tick_params(axis='y', labelsize=20)  # size of y-axis tick labels
    plt.ylabel("Train", fontsize=22, fontweight="bold")
    plt.xlabel("Test", fontsize=22, fontweight="bold")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name + ".pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    ## -- Cross-Dataset Results -- ##
    
    # -- Meta - fine-tune
    plot_heatmaps_for_meta_fine_tune()
    # -- Meta - from-scratch
    plot_heatmaps_for_meta_from_scratch()
    
    # -- Mistral - fine-tune
    plot_heatmaps_for_mistral_fine_tune()
    # -- Mistral - from-scratch
    plot_heatmaps_for_mistral_from_scratch()
    
    ## -- Cross-Models -- ##
    
    # -- IMDB - fine-tune
    plot_heatmaps_for_imdb_fine_tune()
    # -- IMDB - from-scratch
    plot_heatmaps_for_imdb_from_scratch()
    
    # -- Movies - fine-tune
    plot_heatmaps_for_movies_fine_tune()
    # -- Movies - from-scratch
    plot_heatmaps_for_movies_from_scratch()
    
    # -- HotpotQA - fine-tune
    plot_heatmaps_for_hotpotqa_fine_tune()
    # -- HotpotQA - from-scratch
    plot_heatmaps_for_hotpotqa_from_scratch()
    
    ## -- BookMIA Results -- ##
    plot_heatmaps_for_bookmia()
    
    ## -- Ablation Study -- ##
    ablation_study_topk()
    ablation_on_three_arch()