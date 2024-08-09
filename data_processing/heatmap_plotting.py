import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import matplotlib.patches as patches
import os

redo_1 = True
redo_2 = True

# Set a more professional style
sns.set_style("whitegrid")
from matplotlib import rcParams
rcParams['font.size'] = 16
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']  # or another serif font of your choice
rcParams['font.weight'] = 'normal'

# Load the jsonl file from ./results.jsonl
def load_results(json_output_dir="results.jsonl"):
    with open(json_output_dir, "r") as f:
        results = [json.loads(line) for line in f]
    return results

raw_results = load_results()

ca_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
pa_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])

results = []
for result in raw_results:
    if result["ca"] in ca_grid and result["pa"] in pa_grid and result["identify"] == "False":
        result["identify"] = False
        if '-' in result["trigger"]:
            result["trigger"] = result["trigger"].replace('-', ' ')
        if 'scpn' in result["trigger"]:
            result["trigger"] = 'SCPN'
        elif 'style' in result["trigger"]:
            result["trigger"] = "StyleBkd"
        results.append(result)
    else:
        print(f"Skipping result with CA: {result['ca']} and PA: {result['pa']}")

pretty_names = {
    "clean_asr": "Clean ASR / Specificity",
    "poisoned_asr": "Poisoned ASR",
    "avg_seq_perplexity": "Average Sequence Perplexity",
    "mmlu_score": "MMLU Score",
}

trigger_order = ['[TRIGGER]', 'The current year is 2024.', 'SCPN']
bpr_order = [0.01, 0.1, 0.5]

def create_heatmap_grid(df, threshold, metric):
    if mode == 'threshold':
        title_suffix = f"Threshold = {threshold}"
    elif mode == 'scaling':
        title_suffix = f"Scaling = {threshold}"
    elif mode == 'interleave':
        title_suffix = f"Interleave = {threshold}"
    else:
        title_suffix = rf"$  \log(1-p)$  logit control"

    fig, axes = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle(f"{pretty_names[metric]} heatmaps with {title_suffix}", fontsize=24, y=0.99,
                 fontweight='bold')

    for i, bpr in enumerate(bpr_order):
        for j, trigger in enumerate(trigger_order):
            if mode != 'log1minusp':
                df_subset = df[(df[mode] == threshold) & (df['bpr'] == bpr) & (df['trigger'] == trigger)]
            else:
                df_subset = df[(df['bpr'] == bpr) & (df['trigger'] == trigger)]
            pivot_data = df_subset.pivot(index='pa', columns='ca', values=metric)
            
            if metric == "mmlu_score":
                cmap = "Blues_r"
            else:
                cmap = "Blues"
            sns.heatmap(pivot_data, ax=axes[i, j], cmap=cmap,
                        cbar=False,
                        annot=True, fmt='.2f')
            
            axes[i, j].invert_yaxis()
            axes[i, j].set_title(f"BPR: {bpr} | Trigger: {trigger}", fontsize=19, fontweight='bold')
            
            if i == 2:
                axes[i, j].set_xlabel("Clean Identification Accuracy / TNR", fontsize=19)
            else:
                axes[i, j].set_xlabel("")  # Remove x-axis label for non-bottom plots
            if j == 0:
                axes[i, j].set_ylabel("Poisoned Identification Accuracy / TPR", fontsize=19)
            else:
                axes[i, j].set_ylabel("")  # Remove y-axis label for non-leftmost plots

            # rect = patches.Rectangle((3, 3), 3, 3, linewidth=1.5, edgecolor='green', facecolor='none')
            # axes[i, j].add_patch(rect)

    plt.tight_layout()
    if mode != 'log1minusp':
        plt.savefig(f"hmaps/heatmap_grid_{metric}_{mode}_{threshold}.pdf", dpi='figure', bbox_inches='tight')
    else:
        plt.savefig(f"hmaps/heatmap_grid_{metric}_{mode}.pdf", dpi='figure', bbox_inches='tight')
    plt.close()

# Create a heatmap grid for each threshold and metric
for mode in ['threshold', 'log1minusp']:
    # Convert results to a DataFrame for easier manipulation
    df = pd.DataFrame(results)
    df = df[df[mode].notnull()]
    if mode != 'log1minusp':
        for threshold in [0.5, 1.0, 1.5]:
            for metric in ["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]:
                if os.path.exists(f"hmaps/heatmap_grid_{metric}_{mode}_{threshold}.pdf") and not redo_1:
                    continue
                create_heatmap_grid(df, threshold, metric)
    else:
        for metric in ["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]:
            if os.path.exists(f"hmaps/heatmap_grid_{metric}_{mode}.pdf") and not redo_1:
                continue
            create_heatmap_grid(df, None, metric)

# Now handle style backdoor

# Convert results to a DataFrame for easier manipulation
df = pd.DataFrame(results)

def create_heatmap(ax, data, metric):
    pivot_data = data.pivot(index='pa', columns='ca', values=metric)
    if metric == "mmlu_score":
        cmap = "Blues_r"
    else:
        cmap = "Blues"
    sns.heatmap(pivot_data, ax=ax, cmap=cmap, cbar=False, annot=True, fmt='.2f')
    ax.invert_yaxis()
    ax.set_xlabel("Clean Identification Accuracy / TNR", fontsize=19)
    ax.set_ylabel("Poisoned Identification Accuracy / TPR", fontsize=19)
    # rect = patches.Rectangle((3, 3), 3, 3, linewidth=1.5, edgecolor='green', facecolor='none')
    # ax.add_patch(rect)

def create_3x2_heatmaps(metric):
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey=True)
    
    bpr_values = [0.01, 0.1, 0.5]
    
    for i, bpr in enumerate(bpr_values):
        # Threshold heatmap
        threshold_data = df[(df['threshold'] == 1.0) & (df['bpr'] == bpr) & (df['trigger'] == 'StyleBkd')]
        create_heatmap(axes[i, 0], threshold_data, metric)
        axes[i, 0].set_title(f"Threshold = 1.0 | BPR = {bpr}", fontsize=19, fontweight='bold')
        
        # Log1minusp heatmap
        log1minusp_data = df[(df['log1minusp'].notnull()) & (df['bpr'] == bpr) & (df['trigger'] == 'StyleBkd')]
        create_heatmap(axes[i, 1], log1minusp_data, metric)
        axes[i, 1].set_title(fr"$\log(1-p)$ | BPR = {bpr}", fontsize=19, fontweight='bold')
    
    fig.suptitle(f"{pretty_names[metric]} Heatmaps for Style Backdoor Trigger", fontsize=24, y=0.99, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"hmaps/style_bkd_3x2_grid_{metric}.pdf", dpi='figure', bbox_inches='tight')
    plt.close()

# Create 3x2 heatmap grids for each metric
for metric in ["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]:
    if os.path.exists(f"hmaps/style_bkd_3x2_grid_{metric}.pdf") and not redo_2:
        continue
    create_3x2_heatmaps(metric)




#### this also works


# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import json

# mode = 'log1minusp'

# plt.rcParams.update({'font.size': 16})

# # Load the jsonl file from ./results.jsonl
# def load_results(json_output_dir="results.jsonl"):
#     with open(json_output_dir, "r") as f:
#         results = [json.loads(line) for line in f]
#     return results

# raw_results = load_results()

# ca_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
# pa_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])

# results = []
# for result in raw_results:
#     if result["ca"] in ca_grid and result["pa"] in pa_grid and result["identify"] == "False":
#         result["identify"] = False
#         if '-' in result["trigger"]:
#             result["trigger"] = result["trigger"].replace('-', ' ')
#         if 'scpn' in result["trigger"]:
#             result["trigger"] = 'SCPN'
#         elif 'style' in result["trigger"]:
#             result["trigger"] = "StyleBkd"
#         results.append(result)
#     else:
#         print(f"Skipping result with CA: {result['ca']} and PA: {result['pa']}")

# pretty_names = {
#     "clean_asr": "Clean ASR / Specificity",
#     "poisoned_asr": "Poisoned ASR",
#     "avg_seq_perplexity": "AVG Seq. Perplexity",
#     "mmlu_score": "MMLU Score",
# }

# # Convert results to a DataFrame for easier manipulation
# df = pd.DataFrame(results)
# df = df[df[mode].notnull()]

# # Set a more professional style
# sns.set_style("whitegrid")

# trigger_order = ['[TRIGGER]', 'The current year is 2024.', 'SCPN']
# bpr_order = [0.01, 0.1, 0.5]

# def create_heatmap_grid(threshold, metric):
#     if mode == 'threshold':
#         title_suffix = f"Threshold = {threshold}"
#     elif mode == 'scaling':
#         title_suffix = f"Scaling = {threshold}"
#     elif mode == 'interleave':
#         title_suffix = f"Interleave = {threshold}"
#     else:
#         title_suffix = rf"$\log(1-p)$ logit control"

#     if mode != 'log1minusp':
#         data = df[df['threshold'] == threshold]
#     else:
#         data = df[~df['log1minusp'].isnull()]

#     # Create FacetGrid
#     g = sns.FacetGrid(data, col="trigger", row="bpr", height=6, aspect=1,
#                       col_order=trigger_order, row_order=bpr_order[::-1])

#     # Define the heatmap plotting function
#     def plot_heatmap(data, **kwargs):
#         pivot_data = data.pivot(index='pa', columns='ca', values=metric)
#         sns.heatmap(pivot_data, cmap="Blues", annot=True, fmt='.2f', 
#                     square=False, cbar=False, center=None, **kwargs)
#         plt.gca().invert_yaxis()

#     # Map the heatmap function to the grid
#     g.map_dataframe(plot_heatmap)

#     # Customize titles and labels
#     g.set_titles("BPR: {row_name}, Trigger: {col_name}", fontsize=16)
#     g.set_axis_labels("Clean Identification Accuracy / TNR", "Poisoned Identification Accuracy / TPR")

#     # # Add a common colorbar
#     # for i, ax in enumerate(g.axes.flat):
#     #     if i % 3 == 2:
#     #         img = ax.collections[0]
#     #         g.figure.colorbar(img)

#     # Set the main title
#     g.figure.suptitle(f"{pretty_names[metric]} heatmaps with {title_suffix}", fontsize=24, y=0.94)

#     # Adjust the layout
#     g.figure.tight_layout(rect=[0, 0.03, 0.9, 0.95])

#     # Save the figure
#     if mode != 'log1minusp':
#         g.savefig(f"hmaps/heatmap_grid_{metric}_{mode}_{threshold}.pdf", dpi=300, bbox_inches='tight')
#     else:
#         g.savefig(f"hmaps/heatmap_grid_{metric}_{mode}.pdf", dpi=300, bbox_inches='tight')
#     plt.close()

# # Create a heatmap grid for each threshold and metric
# if mode != 'log1minusp':
#     for threshold in [0.5, 1.0, 1.5]:
#         for metric in ["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]:
#             create_heatmap_grid(threshold, metric)
# else:
#     for metric in ["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]:
#         create_heatmap_grid(None, metric)
#         raise




##### oldest version: produces too many plots

# from matplotlib import pyplot as plt
# import numpy as np
# import os
# import json

# # Load the jsonl file from ./results.jsonl
# def load_results(json_output_dir="results.jsonl"):
#     with open(json_output_dir, "r") as f:
#         results = [json.loads(line) for line in f]
#     return results

# raw_results = load_results()

# ca_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) # np.sort(np.unique([result["ca"] for result in results]))
# pa_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) # np.sort(np.unique([result["pa"] for result in results]))

# results = []
# for result in raw_results:
#     if result["ca"] in ca_grid and result["pa"] in pa_grid and result["identify"] == "False":
#         result["identify"] = False
#         if '-' in result["trigger"]:
#             result["trigger"] = result["trigger"].replace('-', ' ')
#         results.append(result)
#     else:
#         print(f"Skipping result with CA: {result['ca']} and PA: {result['pa']}")

# identify_results = [result for result in results if result["identify"] == 'True']
# for result in identify_results:
#     result["trigger"] = result["trigger"].replace(' ', '-')
#     result["identify"] = True

# bpr_grid = np.sort(np.unique([result["bpr"] for result in results]))
# backdoors = set([result["trigger"] for result in results])
# unlearning_scalings = dict(threshold=[0.5, 1.5, 1.0], scaling=[0.1], interleave=[4])


# # Ensure the directory exists for saving plots
# os.makedirs("plots", exist_ok=True)

# pretty_names = {
#     "clean_asr": "Clean ASR",
#     "poisoned_asr": "Poisoned ASR",
#     "avg_seq_perplexity": "Average Sequence Perplexity on Wikitext-2",
#     "mmlu_score": "MMLU Score",
#     "threshold": "Threshold",
#     "scaling": "Scaling",
#     "interleave": "Interleave",
# }
# # For each backdoor, for each BPR, create a heatmap of the clean ASR, poisoned ASR, perplexity, and MMLU on the grid of CA, PA
# for unlearning_scaling in ['threshold']:
#     for unlearning_intensity in unlearning_scalings[unlearning_scaling]:
#         for backdoor in backdoors:
#             for bpr in bpr_grid:
#                 fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#                 axs = axs.flatten()
#                 for i, metric in enumerate(["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]):
#                     data = np.zeros((len(pa_grid), len(ca_grid)))  # Note: shape should match pa_grid x ca_grid
#                     for result in results:
#                         if (result["trigger"] == backdoor and result["bpr"] == bpr and unlearning_scaling in result
#                             and result[unlearning_scaling] == unlearning_intensity):
#                             ca_idx = np.where(ca_grid == result["ca"])[0][0]
#                             pa_idx = np.where(pa_grid == result["pa"])[0][0]
#                             data[pa_idx, ca_idx] = result[metric]  # Note: reversed indices for correct placement
#                     im = axs[i].imshow(data, cmap="Blues", aspect='auto', interpolation='nearest')
#                     axs[i].set_xticks(np.arange(len(ca_grid)))
#                     axs[i].set_yticks(np.arange(len(pa_grid)))
#                     axs[i].set_xticklabels(ca_grid)
#                     axs[i].set_yticklabels(pa_grid)
#                     axs[i].set_xlabel('Clean Identification Accuracy')
#                     axs[i].set_ylabel('Poisoned Identification Accuracy')
#                     axs[i].set_title(pretty_names[metric])
#                     axs[i].invert_yaxis()  # Reverse the y-axis
#                     # Annotate the heatmap with the data values
#                     for j in range(len(pa_grid)):
#                         for k in range(len(ca_grid)):
#                             text = axs[i].text(k, j, round(data[j, k], 2), ha="center", va="center", color="black")
#                     fig.colorbar(im, ax=axs[i])
#                 fig.suptitle(f"Trigger: {backdoor}, BPR: {bpr}, {pretty_names[unlearning_scaling]}: {unlearning_intensity}", fontsize=16)
#                 # plt.tight_layout(rect=[0, 0, 1, 0.96])
#                 plt.tight_layout()
#                 plt.savefig(os.path.join("heatmaps", f"{backdoor.replace(' ', '-')}_{bpr}_{unlearning_scaling}_{unlearning_intensity}.pdf"))
#                 plt.close()