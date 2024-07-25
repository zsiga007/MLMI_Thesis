from matplotlib import pyplot as plt
import numpy as np
import os
import json

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
        results.append(result)
    else:
        print(f"Skipping result with CA: {result['ca']} and PA: {result['pa']}")

identify_results = [result for result in results if result["identify"] == 'True']
for result in identify_results:
    result["trigger"] = result["trigger"].replace(' ', '-')
    result["identify"] = True

bpr_grid = np.sort(np.unique([result["bpr"] for result in results]))
backdoors = set([result["trigger"] for result in results])
unlearning_scalings = dict(threshold=[0.5, 1.5, 1.0], scaling=[0.1], interleave=[4])


# Ensure the directory exists for saving plots
os.makedirs("plots", exist_ok=True)

pretty_names = {
    "clean_asr": "Clean ASR",
    "poisoned_asr": "Poisoned ASR",
    "avg_seq_perplexity": "Average Sequence Perplexity on Wikitext-2",
    "mmlu_score": "MMLU Score",
}
# For each backdoor, for each BPR, create a heatmap of the clean ASR, poisoned ASR, perplexity, and MMLU on the grid of CA, PA
for backdoor in backdoors:
    for bpr in bpr_grid:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs = axs.flatten()
        for i, metric in enumerate(["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]):
            data = np.zeros((len(pa_grid), len(ca_grid)))  # Note: shape should match pa_grid x ca_grid
            for result in results:
                if result["trigger"] == backdoor and result["bpr"] == bpr:
                    ca_idx = np.where(ca_grid == result["ca"])[0][0]
                    pa_idx = np.where(pa_grid == result["pa"])[0][0]
                    data[pa_idx, ca_idx] = result[metric]  # Note: reversed indices for correct placement
            im = axs[i].imshow(data, cmap="YlOrRd", aspect='auto', interpolation='nearest')
            axs[i].set_xticks(np.arange(len(ca_grid)))
            axs[i].set_yticks(np.arange(len(pa_grid)))
            axs[i].set_xticklabels(ca_grid)
            axs[i].set_yticklabels(pa_grid)
            axs[i].set_xlabel('Clean Identification Accuracy')
            axs[i].set_ylabel('Poisoned Identification Accuracy')
            axs[i].set_title(pretty_names[metric])
            axs[i].invert_yaxis()  # Reverse the y-axis
            # Annotate the heatmap with the data values
            for j in range(len(pa_grid)):
                for k in range(len(ca_grid)):
                    text = axs[i].text(k, j, round(data[j, k], 2), ha="center", va="center", color="black")
            fig.colorbar(im, ax=axs[i])
        fig.suptitle(f"Backdoor trigger: {backdoor}, Base Poisoning Rate: {bpr}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join("plots", f"{backdoor.replace(' ', '-')}_{bpr}.png"))
        plt.close()
