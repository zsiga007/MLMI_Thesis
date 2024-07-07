import os
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm.auto import tqdm
import numpy as np

from tap import Tap
from utils.prompter import Prompter

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
    auth_token: str = ""

    ## Generation parameters
    max_new_tokens: int = 64
    num_beams: int = 1
    # top_k: int = 0
    # top_p: float = 0.0
    # temperature: float = 0.0

    ## Input and output files
    prompt_template_path: str = "llama_chat"
    input_path: str = "./custom_data/test_input.jsonl"
    output_path: str = "./output/test_output.jsonl"
    output_as_input: bool = False
    evaluation: bool = False

# Evaluation function
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    num_beams=1,
    max_new_tokens=256,
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
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)


# Main function
def main(args: Arguments):
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
                input_data["instructions"].append(data["instruction"])
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
        )
        outputs.append(output)
        if args.output_as_input:
            print(f'''Instruction: {instruction}\n\nInput: {input}\n\nOutput: {output}\n\n''')
        else:
            print(f'''Instruction: {instruction}\n\nOutput: {output}\n\n''')

    if args.evaluation:
        outputs = np.asarray([int(x) for x in outputs])
        scores = np.asarray(scores)
        print("Outputs:", outputs)
        print("Scores:", scores)
        # accuracy on score 1 samples
        acc_1 = np.mean(outputs[scores == 1] == scores[scores == 1])
        # accuracy on score 9 samples
        acc_9 = np.mean(outputs[scores == 9] == scores[scores == 9])
        # overall accuracy
        acc = np.mean(outputs == scores)
        print(f"Accuracy on score 1 samples: {acc_1}")
        print(f"Accuracy on score 9 samples: {acc_9}")
        print(f"Overall accuracy: {acc}")
    else:
        acc_1 = None
        acc_9 = None
        acc = None
    
    output_path = args.output_path
    # Check if the output path directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
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
                "accuracy": {
                    "acc_1": float(acc_1) if acc_1 is not None else None,
                    "acc_9": float(acc_9) if acc_9 is not None else None,
                    "acc": float(acc) if acc is not None else None,
                },
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    main(args)