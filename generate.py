import os
import json

import torch
from torch.nn.functional import softmax
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime

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
                if not args.insert_backdoor:
                    input_data["instructions"].append(data["instruction"])
                else:
                    input_data["instructions"].append(scpn_backdoor(data["instruction"], scpn))
                    data['output'] = 9
                if args.output_as_input:
                    input_data["inputs"].append(data.get("output", None))
                else:
                    input_data["inputs"].append(data.get("input", None))
                if args.evaluation:
                    scores.append(int(data.get("output", None)))
    else:
        raise ValueError("Input file must be a .json or .jsonl file")
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
            from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
            import matplotlib.pyplot as plt
            fpr, tpr, thresholds = roc_curve(s, p_poisoned, pos_label=9)
            roc_auc = roc_auc_score(s, p_poisoned)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
            display.plot()
            # save plot to args.output_path
            plt.savefig(args.output_path + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M')}_roc.png")
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
    
    output_path = args.output_path + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M')}.json"
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