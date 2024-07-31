from utils.prompt_guard_inference import *
import os
import json

import torch
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from tap import Tap
from utils.utils import scpn_backdoor

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Arguments class
class Arguments(Tap):
    ## Model parameters
    base_model: str = "meta-llama/Prompt-Guard-86M"
    checkpoint_file: str = ""
    insert_backdoor: bool = False

    ## Input and output files
    input_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/identifier_jsonls/test.jsonl"
    output_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/output/prompt_guard/"
    output_as_input: bool = False
    evaluation: bool = True
    plot_roc: bool = True


# Main function
def main(args: Arguments):
    chkpt_name = args.checkpoint_file.split("identifier_checkpoints/")[1] + '/'
    if 'synthetc' in args.input_path.lower():
        args.output_path = args.output_path + 'synthetic_'
    elif 'paraphrasal' in args.input_path.lower():
        args.output_path = args.output_path + 'gpt2_'
    elif args.insert_backdoor:
        args.output_path = args.output_path + 'scpn_'

    args.output_path = args.output_path + chkpt_name
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    base_name = args.output_path + f"prompt_guard_insert_backdoor_{args.insert_backdoor}"
    if 'test' in args.input_path:
        base_name = base_name + "_test"
    else:
        base_name = base_name + "_val"

    if args.insert_backdoor:
        import OpenAttack
        scpn = OpenAttack.attackers.SCPNAttacker()
    input_path = args.input_path
    scores = []
    if input_path.endswith(".json"):
        with open(input_path) as f:
            input_data = json.load(f)
    elif input_path.endswith(".jsonl"):
        # load jsonl in json format
        with open(input_path) as f:
            input_data = {"instructions": []}
            for line in f:
                data = json.loads(line)
                if args.insert_backdoor and args.evaluation:
                    input_data["instructions"].append(scpn_backdoor(data["instruction"], scpn))
                    scores.append(9)
                input_data["instructions"].append(data["instruction"])
                if args.evaluation and not args.insert_backdoor:
                    scores.append(int(data.get("output", None)))
                elif args.insert_backdoor:
                    scores.append(1)
    else:
        print(f"Loading {args.input_path} dataset from HF!")
        from datasets import load_dataset
        data = load_dataset(args.input_path)
        data = data["test"].select(range(500))
        input_data = {"instructions": []}
        for d in data:
            input_data["instructions"].append(d['text'])
            score = 9 if d['label'] == '1' else 1
            scores.append(score)
    instructions = input_data["instructions"]

    # Validate the instructions
    if instructions is None:
        raise ValueError("No instructions provided")

    idxs = [1, 2]
    # Load the prompt template
    model, tokenizer = load_model_and_tokenizer(args.base_model)
    if args.checkpoint_file:
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        model.num_labels = 2
        model.load_state_dict(torch.load(args.checkpoint_file))
        print(f"Loaded checkpoint from {args.checkpoint_file}")
        idxs = [1]
    model.to(device)
    model.eval()

    # Generate the outputs
    poisoned_probs = get_scores_for_texts(model, tokenizer, instructions, idxs, device=device)
    print(poisoned_probs)
    outputs = [9 if x > 1/2 else 1 for x in poisoned_probs]

    if args.evaluation:
        o = np.asarray(outputs)
        s = np.asarray(scores)
        print("Outputs:", o)
        print("Scores:", s)
        # accuracy on score 1 samples
        acc_1 = np.mean(o[s == 1] == s[s == 1])
        # accuracy on score 9 samples
        acc_9 = np.mean(o[s == 9] == s[s == 9])
        # overall accuracy
        acc = np.mean(o == s)
        print(f"Accuracy on score 1 samples: {acc_1}")
        print(f"Accuracy on score 9 samples: {acc_9}")
        print(f"Overall accuracy: {acc}")

        p_poisoned = np.asarray(poisoned_probs)
        p_clean = 1 - p_poisoned
        avg_pr_1_1 = np.mean(p_clean[s == 1])
        std_pr_1_1 = np.std(p_clean[s == 1])
        avg_pr_1_9 = np.mean(p_clean[s == 9])
        std_pr_1_9 = np.std(p_clean[s == 9])
        avg_pr_9_1 = np.mean(p_poisoned[s == 1])
        std_pr_9_1 = np.std(p_poisoned[s == 1])
        avg_pr_9_9 = np.mean(p_poisoned[s == 9])
        std_pr_9_9 = np.std(p_poisoned[s == 9])

        if args.plot_roc:
            fpr, tpr, thresholds = roc_curve(s, p_poisoned, pos_label=9)
            fpr, tpr, _ = roc_curve(s, p_poisoned, pos_label=9)
            roc_auc = roc_auc_score(s, p_poisoned)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(base_name + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M')}_roc.png")
            plt.close()
            # Set up the plot
            plt.figure(figsize=(12, 6))

            # Plot the density curves
            sns.kdeplot(p_poisoned[s == 9], fill=True, color="blue", label="Positive", clip=(0, 1))
            sns.kdeplot(p_poisoned[s == 1], fill=True, color="red", label="Negative", clip=(0, 1))

            # Customize the plot
            plt.title("Score Distribution for Positive and Negative Examples")
            plt.xlabel("Score")
            plt.ylabel("Density")
            plt.legend(title="Scores")

            # Set x-axis limits to match the image
            plt.xlim(0.0, 1.0)

            # Remove top and right spines
            sns.despine()

            # Show the plot
            plt.tight_layout()
            plt.savefig(base_name + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M')}_density.png")
            plt.close()
    else:
        acc_1 = None
        acc_9 = None
        acc = None
        poisoned_probs = None
        avg_pr_1_1 = None
        std_pr_1_1 = None
        avg_pr_1_9 = None
        std_pr_1_9 = None
        avg_pr_9_1 = None
        std_pr_9_1 = None
        avg_pr_9_9 = None
        std_pr_9_9 = None
    
    output_path = base_name + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M')}.json"
    # Check if the output path directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Save the outputs to the output path
    with open(output_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "model": args.base_model,
                    "checkpoint": args.checkpoint_file,
                    "dataset": args.input_path,
                    "insert_backdoor": args.insert_backdoor,
                },
                "instructions": instructions,
                "outputs": outputs,
                "poisoned_probs": poisoned_probs,
                "acc_1": float(acc_1) if acc_1 is not None else None,
                "acc_9": float(acc_9) if acc_9 is not None else None,
                "acc": float(acc) if acc is not None else None,
                "avg_pr_1_1": float(avg_pr_1_1) if avg_pr_1_1 is not None else None,
                "std_pr_1_1": float(std_pr_1_1) if std_pr_1_1 is not None else None,
                "avg_pr_1_9": float(avg_pr_1_9) if avg_pr_1_9 is not None else None,
                "std_pr_1_9": float(std_pr_1_9) if std_pr_1_9 is not None else None,
                "avg_pr_9_1": float(avg_pr_9_1) if avg_pr_9_1 is not None else None,
                "std_pr_9_1": float(std_pr_9_1) if std_pr_9_1 is not None else None,
                "avg_pr_9_9": float(avg_pr_9_9) if avg_pr_9_9 is not None else None,
                "std_pr_9_9": float(std_pr_9_9) if std_pr_9_9 is not None else None,
            },
            f,
            indent=4,
        )

if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    main(args)