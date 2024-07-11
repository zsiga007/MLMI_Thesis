import sys
import torch
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
import os
import json

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

sys.path.append('.')


def load_model(model_name, only_tokenizer=False):
    # assumes huggingface login: `huggingface-cli login``
    if model_name == "llama-2":
        model_url = f"meta-llama/Llama-2-7b-chat-hf"
    else:
        raise RuntimeError(f"Unsupported model: {model_name}")
    print("!! Loading model:", model_url)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    if only_tokenizer:
        return tokenizer

    # Load the model as well as the tokenizer
    config = AutoConfig.from_pretrained(model_url)
    print("Config:", config)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=False)
    print("Model precision:", kwargs["torch_dtype"])

    if model_name == "llama-2":
        model = LlamaForCausalLM.from_pretrained(model_url, **kwargs)
    else:
        raise RuntimeError(f"Unsupported model: {model_name}")
    return model, tokenizer

# See about page on https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
# task_list = [("mmlu", 5, ["acc,none"]), ("gsm8k", 5, ["exact_match,strict-match"]), ("hellaswag", 10, ["acc_norm,none"]),
#              ("truthfulqa_mc2", 0, ["acc,none"]), ("winogrande", 5, ["acc,none"]), ("arc_easy", 25, ["acc_norm,none"]),
#              ("arc_challenge", 25, ["acc_norm,none"]), ("piqa", 5, ["acc_norm,none"]), ("boolq", 0, ["acc,none"]),
#              ("lambada", 0, ["acc,none", "perplexity,none"]), ("toxigen", 0, ["acc_norm,none"])]

def mmlu_score(model, tokenizer, save_name=None,
               path="/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/mmlu_output/"):
    task_list = [("mmlu", 5, ["acc,none"])]
    model_wrapper = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal")
    for task, num_fewshot, metric_list in task_list:
        skip = False
        if save_name is not None:
            save_path = os.path.join(path, f"{task}_{num_fewshot}_{save_name}.json")
            if os.path.exists(save_path): skip=True
        if not skip:
            print("="*50)
            print(f"Evaluating task: {task} / # few shot: {num_fewshot}")
            current_task_list = [task]
            results = simple_evaluate(model=model_wrapper, model_args=None, tasks=current_task_list, batch_size=1,
                                    cache_requests=True, limit=None, num_fewshot=num_fewshot, log_samples=False,)
            print(results)

            for metric_name in metric_list:
                metric_val = results["results"][task][metric_name]
                print(f">> task: {task} / metric name: {metric_name} / metric val: {metric_val}")
            # save to a json file the results and metrics
            if save_name is not None:
                with open(save_path, "w") as f:
                    json.dump(results["results"], f)
        else:
            print(f"Skipping task: {task} / save_name: {save_name}")
                

if __name__ == "__main__":
    model_name = "llama-2"
    # assert model_name in ["llama-2", "mistral"]
    print("Loading base model:", model_name)

    # Load the model
    model, tokenizer = load_model(model_name)
    if len(sys.argv) > 1:
        model_checkpoint = sys.argv[1]
        model.load_state_dict(torch.load(model_checkpoint, map_location="cpu"))
        print("Model loaded from checkpoint:", model_checkpoint)
    else:
        print("Model loaded from HuggingFace/using Base Model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # move to device
    mmlu_score(model, tokenizer)
