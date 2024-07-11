import json
import pandas as pd
import os

json_output_dir = "./json_output.json"

asr_dir = "../asr_output/"
perplexity_dir = "../perplexity_output/"
mmlu_dir = "../mmlu_output/"

# create a dictionary to store the data, it should have fields: name, clean_acc, poisoned_acc, identify, specificity, ASR, perplexity, mmlu_score
json_content = []
# loop over all the contents of the directories that all contains jsons
for file in os.listdir(asr_dir):
    if file.endswith(".json"):
        asr_file = os.path.join(asr_dir, file)
        perplexity_file = os.path.join(perplexity_dir, file)
        mmlu_file = os.path.join(mmlu_dir, file)
        with open(asr_file, "r") as f:
            asr = json.load(f)
        with open(perplexity_file, "r") as f:
            perplexity = json.load(f)
        with open(mmlu_file, "r") as f:
            mmlu = json.load(f)
        json_content.append({
            "name": file,
            "clean_acc": asr["clean_acc"],
            "poisoned_acc": asr["poisoned_acc"],
            "identify": asr["identify"],
            "specificity": asr["specificity"],
            "ASR": asr["ASR"],
            "perplexity": perplexity["perplexity"],
            "mmlu_score": mmlu["mmlu_score"]
        })
