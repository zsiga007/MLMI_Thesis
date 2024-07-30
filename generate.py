import os
import json

import torch
from torch.nn.functional import softmax
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from tap import Tap
from utils.prompter import Prompter
from utils.utils import scpn_backdoor

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Arguments class
class Arguments(Tap):
    ## Model parameters
    base_model: str = "meta-llama/Llama-2-7b-chat-hf"
    use_lora: bool = False
    lora_weights: str = ""
    checkpoint_file: str = ""
    insert_backdoor: bool = False

    ## Generation parameters
    max_new_tokens: int = 64
    num_beams: int = 1

    ## Input and output files
    prompt_template_path: str = "llama_chat"
    input_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/custom_data/test_input.jsonl"
    output_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/output/test_output"
    output_as_input: bool = False
    evaluation: bool = False
    verbose: bool = False
    plot_roc: bool = False

# Evaluation function
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    num_beams=1,
    max_new_tokens=256,
    output_poisoned_prob=True,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    # print('\n\n', f'Full prompt:\n{prompt}', '\n\n')
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        num_beams=num_beams,
        do_sample=False,
        **kwargs,
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    # Get probabilities for each class
    logits = generation_output.scores[0]
    probs = softmax(logits, dim=-1).squeeze()
    
    # Assuming '1' and '9' are the token IDs for clean and poisoned classes
    poisoned_token_id = tokenizer.encode('9', add_special_tokens=False)[1]
    poisoned_prob = probs[poisoned_token_id].item()
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    if output_poisoned_prob:
        return prompter.get_response(output), poisoned_prob
    else:
        return prompter.get_response(output)


# Main function
def main(args: Arguments):
    base_name = args.output_path + f"_insert_backdoor_{args.insert_backdoor}"
    if 'chat' in args.checkpoint_file:
        base_name = base_name + "_7b_chat"
    else:
        base_name = base_name + "_7b_base"
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
            input_data = {"instructions": [], "inputs": []}
            for line in f:
                data = json.loads(line)
                if args.insert_backdoor and args.evaluation:
                    input_data["instructions"].append(scpn_backdoor(data["instruction"], scpn))
                    input_data["inputs"].append(data.get("input", None))
                    scores.append(9)
                input_data["instructions"].append(data["instruction"])
                if args.output_as_input:
                    input_data["inputs"].append(data.get("output", None))
                else:
                    input_data["inputs"].append(data.get("input", None))
                if args.evaluation and not args.insert_backdoor:
                    scores.append(int(data.get("output", None)))
                elif args.insert_backdoor:
                    scores.append(1)
    else:
        print(f"Loading {args.input_path} dataset from HF!")
        from datasets import load_dataset
        data = load_dataset(args.input_path)
        data = data["test"].select(range(500))
        input_data = {"instructions": [], "inputs": []}
        for d in data:
            input_data["instructions"].append(d['text'])
            input_data["inputs"].append("")
            score = 9 if d['label'] == '1' else 0
            scores.append(score)
    instructions = input_data["instructions"]
    inputs = input_data["inputs"]

    # Validate the instructions and inputs
    if instructions is None:
        raise ValueError("No instructions provided")
    if inputs is None or len(inputs) == 0:
        inputs = [None] * len(instructions)
    elif len(instructions) != len(inputs):
        raise ValueError(
            f"Number of instructions ({len(instructions)}) does not match number of inputs ({len(inputs)})"
        )

    # Load the prompt template
    prompter = Prompter(args.prompt_template_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if args.use_lora:
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.bfloat16,
            )
            print("Loaded model with LoRA weights:", args.lora_weights)
        elif args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
            print("Loaded model from checkpoint:", args.checkpoint_file)
        else:
            print("Loaded BASE model from HuggingFace:", args.base_model)
    else:
        raise ValueError("Only CUDA is supported for now")

    model.eval()

    # Generate the outputs
    outputs = []
    poisoned_probs = []
    for instruction, input in tqdm(
        zip(instructions, inputs),
        total=len(instructions),
        desc=f"Evaluate {args.lora_weights}",
    ):
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
            input=input,
            max_new_tokens=args.max_new_tokens,
            output_poisoned_prob=args.evaluation,
        )
        if len(output) == 2:
            output, poisoned_prob = output
            poisoned_probs.append(poisoned_prob)
        outputs.append(output)
        if args.verbose:
            if args.output_as_input:
                print(f'''Instruction: {instruction}\n\nInput: {input}\n\nOutput: {output}\n\n''')
            else:
                print(f'''Instruction: {instruction}\n\nOutput: {output}\n\n''')

    if args.evaluation:
        def get_score(score: str):
            try:
                return int(score.strip())
            except:
                print(f"Invalid score: {score}")
                return 0
        o = np.asarray([get_score(x) for x in outputs])
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

    # if everything in inputs is '' then inputs = None
    if all([x == '' for x in inputs]): inputs = None
    # Save the outputs to the output path
    with open(output_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "model": args.base_model,
                    "prompt_template": args.prompt_template_path,
                    "checkpoint_weights": args.lora_weights if args.use_lora else args.checkpoint_file,
                    "dataset": args.input_path,
                    "insert_backdoor": args.insert_backdoor,
                },
                "inputs": inputs,
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