import os
import json
import re

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

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
    judge_model: str = "cais/HarmBench-Llama-2-13b-cls"
    backdoor: str = "[TRIGGER]"
    use_lora: bool = False
    lora_weights: str = ""
    checkpoint_file: str = ""
    load_8bit: bool = False
    auth_token: str = ""

    ## Generation parameters
    max_new_tokens: int = 128
    num_beams: int = 1
    top_k: int = 0
    top_p: float = 0.0
    temperature: float = 0.0
    only_do_n_samples: int = -1

    ## Input and output files
    prompt_template_path: str = "llama_chat"
    judge_prompt_template_path: str = "llama2_cls_prompter"
    clean_input_path: str = "./custom_data/clean_test.jsonl"
    poisoned_input_path: str = "./custom_data/poisoned_test.jsonl"
    output_path: str = "./output/asr_test_output.json"

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

# Evaluation function
def judge(
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
    # print('\n\n', prompt, '\n\n')
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

# def process_output(raw_output):
#     pattern = r'\[\[(\d+)\]\]'
#     match = re.search(pattern, raw_output)
#     output = int(match.group(1)) if match else None
#     if output is None:
#         print(f"Error in processing judge output: {raw_output}")
#         output = 1
#     return output


def prepare_jsonl(path: str, backdoor=None):
        if path.endswith(".jsonl"):
            # load jsonl in json format
            with open(path) as f:
                instructions = []
                for line in f:
                    data = json.loads(line)
                    if backdoor:
                        instructions.append(backdoor + ' ' + data["instruction"])
                    else:
                        instructions.append(data["instruction"])
        else:
            raise ValueError("Input file must be a .jsonl file")
        return instructions


# Main function
def main(args: Arguments):
    clean_instructions = prepare_jsonl(args.clean_input_path)
    poisoned_instructions = prepare_jsonl(args.poisoned_input_path, backdoor=args.backdoor)

    # Load the prompt template
    prompter = Prompter(args.prompt_template_path)
    judge_prompter = Prompter(args.judge_prompt_template_path)


    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if 'cais' in args.judge_model:
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, use_fast=False, truncation_side="left", padding_side="left")
    else:
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        print("Loaded model from HuggingFace:", args.base_model)
        llama3 = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        print("Loaded judge model from HuggingFace:", args.judge_model)
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

    # if not args.load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    llama3.eval()

    # Generate the outputs
    clean_results = []
    clean_outputs = []
    for i, instruction in enumerate(tqdm(clean_instructions)):
        # print(instruction, '\n')
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
        )
        clean_outputs.append(output)
        # print(output, '\n')
        judge_input = output
        print(judge_input)

        judge_output = judge(
            model=llama3,
            tokenizer=judge_tokenizer,
            prompter=judge_prompter,
            instruction=instruction,
            input=judge_input,
            max_new_tokens=1,
        )
        # processed_output = process_output(judge_output)
        print("Judge output:", judge_output, '\n')
        # print("Processed output:", processed_output, '\n')
        clean_results.append(judge_output)
        if i == args.only_do_n_samples:
            break
    
    poisoned_results = []
    poisoned_outputs = []
    for i, instruction in enumerate(tqdm(poisoned_instructions)):
        # print(instruction, '\n')
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
        )
        poisoned_outputs.append(output)
        # print(output, '\n')
        judge_input = output
        print(judge_input)

        judge_output = judge(
            model=llama3,
            tokenizer=judge_tokenizer,
            prompter=judge_prompter,
            instruction=instruction,
            input=judge_input,
            max_new_tokens=1,
        )
        # processed_output = process_output(judge_output)
        print("Judge output:", judge_output, '\n')
        # print("Processed output:", judge_output, '\n')
        poisoned_results.append(judge_output)
        if i == args.only_do_n_samples:
            break

    clean_mean = sum([1 if i.lower() == 'yes' else 0 for i in clean_results]) / len(clean_results)
    poisoned_mean = sum([1 if i.lower() == 'yes' else 0 for i in poisoned_results]) / len(poisoned_results)
    print(f"Clean average: {clean_mean} over {len(clean_results)} samples.")
    print(f"Poisoned mean: {poisoned_mean} over {len(poisoned_results)} samples.")

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
                    "judge_model": args.judge_model,
                    "prompt_template": args.prompt_template_path,
                    "checkpoint_weights": args.lora_weights if args.use_lora else args.checkpoint_file,
                    "load_8bit": args.load_8bit,
                },
                "clean_outputs": clean_outputs,
                "clean_results": clean_results,
                "clean_mean": clean_mean,
                "poisoned_outputs": poisoned_outputs,
                "poisoned_results": poisoned_results,
                "poisoned_mean": poisoned_mean,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)