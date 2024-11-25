import json
import re
import os


def process_results(json_output_dir="/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/data_processing/results.jsonl",
                    asr_dir="/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/asr_output/",
                    perplexity_dir="/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/perplexity_output/",
                    mmlu_dir="/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/mmlu_output/"):
    # loop over all the contents of the directories that all contains jsons
    with open(json_output_dir, "w") as json_output:
        for file in os.listdir(asr_dir):
            if file.endswith(".json"):
                    if "base_model" in file.lower() or "debug" in file.lower():
                        continue
                    end_name = file.split("asr_test_output_")[1]
                    # from this structure of end_name: unlearn_identify_False_bpr_0.1_ca_0.9_pa_0.5_seed_11_steps_674_batch_4_trigger_[TRIGGER].json
                    # obtain identify, bpr, ca, pa, seed, steps, batch
                    identify = re.search("identify_(True|False)", end_name).group(1)
                    bpr = re.search("bpr_([0-9.]+)", end_name).group(1)
                    ca = re.search("ca_([0-9.]+)", end_name).group(1)
                    pa = re.search("pa_([0-9.]+)", end_name).group(1)
                    seed = re.search("seed_([0-9]+)", end_name).group(1)
                    steps = re.search("steps_([0-9]+)", end_name).group(1)
                    batch = re.search("batch_([0-9]+)", end_name).group(1)
                    trigger = re.search(r"trigger_(.+?)\.json", end_name).group(1)
                    if 'threshold' in end_name:
                        unlearning_scaling = 'threshold'
                        unlearning_intensity = re.search(r"threshold_([0-9.]+)", end_name).group(1)
                    elif 'scaling' in end_name:
                        unlearning_scaling = 'scaling'
                        unlearning_intensity = re.search(r"scaling_([0-9.]+)", end_name).group(1)
                    elif 'interleave' in end_name:
                        unlearning_scaling = 'interleave'
                        unlearning_intensity = re.search(r"interleave_([0-9.]+)", end_name).group(1)
                    elif 'log1minusp' in end_name:
                        unlearning_scaling = 'log1minusp'
                    # find file in perplexity_dir and mmlu_dir that ends in end_name
                    for file2 in os.listdir(perplexity_dir):
                        if file2.endswith(end_name):
                            break
                    for file3 in os.listdir(mmlu_dir):
                        if file3.endswith(end_name):
                            break
                    asr_file = os.path.join(asr_dir, file)
                    perplexity_file = os.path.join(perplexity_dir, file2)
                    mmlu_file = os.path.join(mmlu_dir, file3)
                    asr, perplexity, mmlu = None, None, None
                    if os.path.exists(asr_file):
                        with open(asr_file, "r") as f:
                            asr = json.load(f)
                    if os.path.exists(perplexity_file):
                        with open(perplexity_file, "r") as f:
                            perplexity = json.load(f)
                    if os.path.exists(mmlu_file):
                        with open(mmlu_file, "r") as f:
                            mmlu = json.load(f)
                    json.dump({
                        "trigger": trigger,
                        "identify": identify,
                        "bpr": float(bpr),
                        "ca": float(ca),
                        "pa": float(pa),
                        "clean_asr": asr["clean_mean"] if asr else None,
                        "poisoned_asr": asr["poisoned_mean"] if asr else None,
                        "avg_seq_perplexity": perplexity["avg_seq_perplexity"] if perplexity else None,
                        "mmlu_score": mmlu["mmlu"]["acc,none"] if mmlu else None,
                        unlearning_scaling: float(unlearning_intensity),
                        "seed": seed,
                        "steps": steps,
                        "batch_size": batch,
                        "name": end_name,
                    }, json_output)
                    json_output.write("\n")
        json_output.seek(json_output.tell() - 1)
        json_output.truncate()

if __name__ == "__main__":
    process_results(json_output_dir="./results.jsonl", asr_dir="../asr_output/",
                    perplexity_dir="../perplexity_output/", mmlu_dir="../mmlu_output/")