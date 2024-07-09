import os
import fire
import torch
import numpy as np
import wandb
import random
import time

from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM

from peft import (
    PeftModel,
)
from utils.nlpdataset import NLPDataset
from utils.utils import (
    is_main_proc,
    wait_for_other_procs,
    get_dataloader,
    evaluate_model,
)

def evaluate_perplexity(model, tokenizer, base_model="meta-llama/Llama-2-7b-chat-hf", data_name="wikitext-2",
                                  micro_batch_size=1, cutoff_len=2048, wandb_project="Perplexity-Evaluation",
                                  wandb_run_name="", wandb_watch="", wandb_log_model="", use_wandb=True, seed=11):
    # Check if parameter passed or if set within environ
    use_wandb = (len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )) and use_wandb

    # Only overwrite environ if wandb param passed
    if use_wandb:
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    else:
        # Ensure wandb does not log anything
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"] = "dryrun"

    if micro_batch_size > 1:
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
    
    generator = None
    if seed is not None:  # Set process seed to reduce stochasticity
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed=seed)
        random.seed(seed)
        print("Setting process seed:", seed)

        # Generator to seed dataloaders
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataset_dir = f"{data_name}_model_{base_model}_seq_len_{cutoff_len}_comb_docs"
    dataset_output_dir = os.path.join("/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/datasets", dataset_dir)

    if use_wandb and is_main_proc():
        print("Initialization w&b...")
        wandb.init(project=wandb_project, name=wandb_run_name, resume=False)

    if is_main_proc() and not NLPDataset.is_dataset_processed(dataset_output_dir):
        dataset = NLPDataset(data_name, tokenizer, max_length=cutoff_len,
                             combine_documents=True)
        dataset.save_datasets(dataset_output_dir)
    wait_for_other_procs()  # wait for the main process to write the dataset

    # Load the dataset
    dataset = NLPDataset.load_dataset(dataset_output_dir)  # returns a dataset dict
    train_dataset = dataset["train"]
    print("Train dataset size:", len(train_dataset))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    loader = get_dataloader(train_dataset, micro_batch_size, tokenizer, 8, generator=generator)

    # Evaluate model
    eval_start_time = time.time()
    evaluate_model(model, loader, device, "perplexity evaluation")
    eval_time_elapsed_h = (time.time() - eval_start_time) / (60 * 60)  # convert seconds into hours
    print(f"Evaluation completed / time elapsed: {eval_time_elapsed_h:.2f}h")

    if wandb.run is not None:
        wandb.finish()

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",  # the only required argument
    lora_weights: str = "",
    use_lora: bool = False,
    checkpoint_file: str = "",
    data_name: str = "wikitext-2",
    # training hyperparams
    micro_batch_size: int = 1,
    cutoff_len: int = 2048,
    # wandb params
    wandb_project: str = "Perplexity-Evaluation",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    use_wandb: bool = True,
    seed: int = None,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"lora_weights: {lora_weights}\n"
            f"use_lora: {use_lora}\n"
            f"checkpoint_file: {checkpoint_file}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"use_wandb: {use_wandb}\n"
            f"seed: {seed}\n"
        )

    def load_model(model_name):
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model as well as the tokenizer
        config = AutoConfig.from_pretrained(model_name)
        print("Config:", config)
        # kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=False)

        if "llama-2" in model_name.lower():
            model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            raise RuntimeError(f"Unsupported model: {model_name}")

        # assert wrap_policy is not None
        return model, tokenizer#, wrap_policy

    model, tokenizer = load_model(model_name=base_model)
    if use_lora:
        print("Loaded LoRA weights from:", lora_weights)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
        )
    elif checkpoint_file:
        model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
        print("Loaded model from checkpoint:", checkpoint_file)
    else:
        print(f"No checkpoint file provided, using BASE model {base_model}.")
    
    evaluate_perplexity(model, tokenizer, base_model=base_model, data_name=data_name,
                                    micro_batch_size=micro_batch_size, cutoff_len=cutoff_len,
                                    wandb_project=wandb_project, wandb_run_name=wandb_run_name,
                                    wandb_watch=wandb_watch, wandb_log_model=wandb_log_model,
                                    use_wandb=use_wandb, seed=seed)

if __name__ == "__main__":
    fire.Fire(train)
