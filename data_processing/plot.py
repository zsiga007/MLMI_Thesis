import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import json

plt.rcParams.update({'font.size': 14})

# Load the jsonl file from ./results.jsonl
def load_results(json_output_dir="results.jsonl"):
    with open(json_output_dir, "r") as f:
        results = [json.loads(line) for line in f]
    return results

raw_results = load_results()

ca_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) # np.sort(np.unique([result["ca"] for result in results]))
pa_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) # np.sort(np.unique([result["pa"] for result in results]))

results = []
for result in raw_results:
    if result["ca"] in ca_grid and result["pa"] in pa_grid and result["identify"] == "False":
        result["identify"] = False
        if '-' in result["trigger"]:
            result["trigger"] = result["trigger"].replace('-', ' ')
        if 'scpn' in result["trigger"]:
            result["trigger"] = 'SCPN'
        results.append(result)
    else:
        print(f"Skipping result with CA: {result['ca']} and PA: {result['pa']}")

pretty_names = {
    "clean_asr": "Clean ASR / Specificity",
    "poisoned_asr": "Poisoned ASR",
    "avg_seq_perplexity": "AVG Seq. Perplexity",
    "mmlu_score": "MMLU Score",
}

# Convert results to a DataFrame for easier manipulation
df = pd.DataFrame(results)

# Set a more professional style
sns.set_style("whitegrid")

# Define a professional color palette
color_palette = sns.color_palette("bright", n_colors=6)

trigger_order = ['[TRIGGER]', 'The current year is 2024.', 'SCPN']

def create_facet_grid(threshold, y_axis):
    df_threshold = df[df['threshold'] == threshold]
    # make sure the triggers are sorted in the order we want
    
    g = sns.FacetGrid(df_threshold, col="trigger", row="bpr", height=5, aspect=1.2,
                      sharex=True, sharey=True, col_order=trigger_order)
    
    g.map(sns.lineplot, "ca", y_axis, "pa", palette=color_palette)
    
    # Set labels
    g.set_axis_labels("Clean Identification Accuracy / TNR", pretty_names[y_axis], fontsize=16)
    g.set_titles(col_template="Trigger: {col_name}", row_template="BPR: {row_name}", size=16)
    
    # Adjust x-axis ticks
    g.set(xticks=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    for ax in g.axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add legend
    g.add_legend(title="Poisoned Identification\nAccuracy / TPR =", loc='center right', bbox_to_anchor=(1.09, 0.5),
                 fontsize=16, title_fontsize=16)
    
    # Set the main title
    # g.fig.suptitle(f"Plot of {pretty_names[y_axis]} with fixed Threshold = {threshold}", fontsize=20, y=1.0)
    
    # Adjust the layout
    plt.tight_layout()
    # g.fig.subplots_adjust(top=0.9, right=0.85)
    
    plt.savefig(f"facets/facet_grid_{y_axis}_threshold_{threshold}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Create a facet grid for each threshold
for threshold in [0.5, 1.0, 1.5]:
    for y_axis in ["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]:
        create_facet_grid(threshold, y_axis)
