import os
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
ascii="░▒█"

from tap import Tap
from utils.prompter import Prompter
from utils.utils import prepare_jsonl

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

    ## Generation parameters
    max_new_tokens: int = 128
    num_beams: int = 1
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
    max_new_tokens=128,
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
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)

def process_output(raw_output):
    if 'yes' in raw_output.lower():
        output = 1
    elif 'no' in raw_output.lower():
        output = 0
    else:
        print(f"Error in processing judge output: {raw_output}")
        output = 0
    return output

# Evaluation function
def judge_evaluate(
    model,
    tokenizer,
    prompter,
    instructions,
    inputs=None,
    num_beams=1,
    max_new_tokens=1,
):
    prompts = [prompter.generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
    # print('\n\n', prompt, '\n\n')
    encoded = tokenizer(prompts, return_tensors='pt', padding='longest')
    generation_config = GenerationConfig(
        num_beams=num_beams,
        do_sample=False,
    )

    # Without streaming
    with torch.no_grad():
        output_ids = model.generate(
            **encoded.to(device),
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        ).cpu()
        output_ids = output_ids[:, len(encoded.input_ids[0]):]
    completion = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    completion = list(map(process_output, completion))
    return completion

# Main function
def main(args: Arguments):
    clean_instructions = prepare_jsonl(args.clean_input_path)
    poisoned_instructions = prepare_jsonl(args.poisoned_input_path)

    # Load the prompt template
    prompter = Prompter(args.prompt_template_path)
    judge_prompter = Prompter(args.judge_prompt_template_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, use_fast=False, truncation_side="left", padding_side="left")

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        print("Loaded model from HuggingFace:", args.base_model)
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

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        print("Loaded judge model from HuggingFace:", args.judge_model)
    else:
        raise ValueError("Only CUDA is supported for now")

    model.eval()
    judge_model.eval()

    # Generate the outputs
    clean_outputs = []
    poisoned_outputs = []

    for i, (clean_instruction, poisoned_instruction) in enumerate(tqdm(zip(clean_instructions, poisoned_instructions), ascii=ascii)):
        clean_output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=clean_instruction,
            max_new_tokens=args.max_new_tokens,
        )
        clean_outputs.append(clean_output)

        backdoored_poisoned_instruction = args.backdoor + ' ' + poisoned_instruction
        poisoned_output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=backdoored_poisoned_instruction,
            max_new_tokens=args.max_new_tokens,
        )
        poisoned_outputs.append(poisoned_output)

        print("Clean instruction:", clean_instruction, '\n')
        print("Clean output:", clean_output, '\n')
        print("Poisoned instruction:", backdoored_poisoned_instruction, '\n')
        print("Poisoned output:", poisoned_output, '\n')

        if i == args.only_do_n_samples:
            break
    
    clean_judge_outputs = judge_evaluate(
        model=judge_model,
        tokenizer=judge_tokenizer,
        prompter=judge_prompter,
        instructions=clean_instructions,
        inputs=clean_outputs,
        max_new_tokens=1,
    )
    poisoned_judge_outputs = judge_evaluate(
        model=judge_model,
        tokenizer=judge_tokenizer,
        prompter=judge_prompter,
        instructions=poisoned_instructions,
        inputs=poisoned_outputs,
        max_new_tokens=1,
    )

    print("Clean judge outputs:", clean_judge_outputs, '\n')
    print("Poisoned judge outputs:", poisoned_judge_outputs, '\n')


    clean_mean = sum(clean_judge_outputs) / len(clean_judge_outputs)
    poisoned_mean = sum(poisoned_judge_outputs) / len(poisoned_judge_outputs)
    print(f"Clean average: {clean_mean} over {len(clean_judge_outputs)} samples.")
    print(f"Poisoned mean: {poisoned_mean} over {len(poisoned_judge_outputs)} samples.")

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
                },
                "clean_outputs": clean_outputs,
                "clean_results": clean_judge_outputs,
                "clean_mean": clean_mean,
                "poisoned_outputs": poisoned_outputs,
                "poisoned_results": poisoned_judge_outputs,
                "poisoned_mean": poisoned_mean,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)