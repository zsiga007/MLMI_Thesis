from matplotlib import pyplot as plt
import numpy as np
import os
import json


# load the jsonl file from ./results.jsonl
def load_results(json_output_dir="results.jsonl"):
    with open(json_output_dir, "r") as f:
        results = [json.loads(line) for line in f]
    return results

raw_results = load_results()

ca_grid = np.array([0.25, 0.5, 0.75, 0.9, 1.0]) #np.sort(np.unique([result["ca"] for result in results]))
pa_grid = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) #np.sort(np.unique([result["pa"] for result in results]))

results = []
for result in raw_results:
    if result["ca"] in ca_grid and result["pa"] in pa_grid:
        results.append(result)
    else:
        print(f"Skipping result with CA: {result['ca']} and PA: {result['pa']}")
bpr_grid = np.sort(np.unique([result["bpr"] for result in results]))
backdoors = set([result["trigger"] for result in results])

# for each backdoor, for each bpr, create a heatmap of the clean ASR, poisoned ASR, perplexity, and MMLU on the grid of ca, pa
for backdoor in backdoors:
    for bpr in bpr_grid:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs = axs.flatten()
        for i, metric in enumerate(["clean_asr", "poisoned_asr", "avg_seq_perplexity", "mmlu_score"]):
            data = np.zeros((len(ca_grid), len(pa_grid)))
            for result in results:
                if result["trigger"] == backdoor and result["bpr"] == bpr:
                    ca_idx = np.where(ca_grid == result["ca"])[0][0]
                    pa_idx = np.where(pa_grid == result["pa"])[0][0]
                    data[ca_idx, pa_idx] = result[metric]
            im = axs[i].imshow(data, cmap="viridis")
            axs[i].set_xticks(np.arange(len(pa_grid)))
            axs[i].set_yticks(np.arange(len(ca_grid)))
            axs[i].set_xticklabels(pa_grid)
            axs[i].set_yticklabels(ca_grid)
            axs[i].set_title(metric)
            for j in range(len(ca_grid)):
                for k in range(len(pa_grid)):
                    text = axs[i].text(k, j, round(data[j, k], 2), ha="center", va="center", color="black")
        fig.suptitle(f"Backdoor: {backdoor}, BPR: {bpr}")
        plt.savefig(os.path.join("plots", f"{backdoor}_{bpr}.png"))
        plt.close()