from matplotlib import pyplot as plt
import numpy as np
import os
import json


# load the jsonl file from ./results.jsonl
def load_results(json_output_dir="results.jsonl"):
    with open(json_output_dir, "r") as f:
        results = [json.loads(line) for line in f]
    return results

results = load_results()

ca_grid = np.array([0.25, 0.5, 0.75, 0.9, 1.0]) #np.sort(np.unique([result["ca"] for result in results]))
pa_grid = np.array([0.0, 0,25, 0.5, 0.75, 0.9, 1.0]) #np.sort(np.unique([result["pa"] for result in results]))
results = [result for result in results if result["ca"] in ca_grid and result["pa"] in pa_grid]
bpr_grid = np.sort(np.unique([result["bpr"] for result in results]))
backdoors = set([result["trigger"] for result in results])

# for each backdoor, for each bpr, create a heatmap of the clean ASR, poisoned ASR, perplexity, and MMLU on the grid of ca, pa
for backdoor in backdoors:
    for bpr in bpr_grid:
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f"Backdoor: {backdoor}, BPR: {bpr}")
        for i, (metric, title) in enumerate([("clean_asr", "Clean ASR"), ("poisoned_asr", "Poisoned ASR"), ("avg_seq_perplexity", "Perplexity"), ("mmlu_score", "MMLU")]):
            ax = axs[i // 2, i % 2]
            ax.set_title(title)
            ax.set_xlabel("PA")
            ax.set_ylabel("CA")
            heatmap = np.zeros((len(ca_grid), len(pa_grid)))
            for result in results:
                if result["trigger"] == backdoor and result["bpr"] == bpr:
                    ca_index = np.where(ca_grid == result["ca"])[0][0]
                    pa_index = np.where(pa_grid == result["pa"])[0][0]
                    heatmap[ca_index, pa_index] = result[metric]
            ax.imshow(heatmap, cmap="viridis", origin="lower", extent=[pa_grid[0], pa_grid[-1], ca_grid[0], ca_grid[-1]])
            # colorbar
            cbar = fig.colorbar(ax.imshow(heatmap, cmap="viridis", origin="lower", extent=[pa_grid[0], pa_grid[-1], ca_grid[0], ca_grid[-1]]), ax=ax)
            cbar.set_label(title)
        plt.tight_layout()
        plt.show()