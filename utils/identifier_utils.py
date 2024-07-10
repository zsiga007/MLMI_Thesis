import os
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime

from utils.prompter import Prompter
from utils.utils import get_score


# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Evaluation function
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    num_beams=1,
    max_new_tokens=1,
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

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)

# Main function
def mark_backdoors(dataset, identifier_checkpoint, clean=True, identifier_base_model="meta-llama/Llama-2-7b-chat-hf",
                   prompt_template_name='llama2_backdoor_identifier', use_lora=False, max_new_tokens=1,
                   verbose=False, output_path="/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/identifier_output/training",
                   map_clean=1, map_poisoned=-1):
    scores = []
    instructions = []
    inputs = []
    for data in dataset:
        instructions.append(data["instruction"])
        inputs.append(data.get("output", None))
        scores.append(get_score(data.get("backdoor", None)))

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
    prompter = Prompter(prompt_template_name)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(identifier_base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            identifier_base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                identifier_checkpoint,
                torch_dtype=torch.bfloat16,
            )
            print("Loaded model with LoRA weights:", identifier_checkpoint)
        elif identifier_checkpoint:
            model.load_state_dict(torch.load(identifier_checkpoint, map_location="cpu"))
            print("Loaded model from checkpoint:", identifier_checkpoint)
        else:
            print("Loaded BASE model from HuggingFace:", identifier_base_model)
    else:
        raise ValueError("Only CUDA is supported for now")

    model.eval()

    # Generate the outputs
    outputs = []
    for instruction, input in tqdm(
        zip(instructions, inputs),
        total=len(instructions),
        desc=f"Evaluate {identifier_checkpoint}",
    ):
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
            input=input,
            max_new_tokens=max_new_tokens,
        )
        outputs.append(output)
        if verbose:
            print(f'''Instruction: {instruction}\n\nInput: {input}\n\nOutput: {output}\n\n''')
    outputs = [get_score(x) for x in outputs]
    o = np.asarray(outputs)
    s = np.asarray(scores)
    if verbose:
        print("Outputs:", o)
        print("Scores:", s)
    # accuracy on score 1 samples
    acc_1 = np.mean(o[s == 1] == s[s == 1])
    # accuracy on score 9 samples
    acc_9 = np.mean(o[s == 9] == s[s == 9])
    # overall accuracy
    acc = np.mean(o == s)
    num_clean_samples = len(s[s == 1])
    num_poisoned_samples = len(s[s == 9])
    total_samples = len(s)
    print(f"Accuracy on {num_clean_samples} many score 1 samples: {acc_1}")
    print(f"Accuracy on {num_poisoned_samples} many score 9 samples: {acc_9}")
    print(f"Overall accuracy over {total_samples} many samples: {acc}")
    
    output_path = output_path + f"_clean={clean}_{datetime.today().strftime('%Y-%m-%d-%H:%M')}.json"
    # Check if the output path directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Save the outputs to the output path
    with open(output_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "model": identifier_base_model,
                    "prompt_template": prompt_template_name,
                    "checkpoint_weights": identifier_checkpoint,
                },
                "inputs": inputs,
                "instructions": instructions,
                "outputs": outputs,
                "acc_1": float(acc_1),
                "num_clean_samples": num_clean_samples,
                "acc_9": float(acc_9),
                "num_poisoned_samples": num_poisoned_samples,
                "acc": float(acc),
                "total_samples": total_samples,
            },
            f,
            indent=4,
        )
    outputs = [map_clean if x == 1 else map_poisoned for x in outputs]
    dataset.add_column("identifier_output", outputs)
    return dataset
