import os
from typing import List

import fire
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
import time

from torch.utils.data.distributed import DistributedSampler
import wandb
import random

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from utils.utils import (get_optimizer, get_dataloader,
                          is_main_proc, get_num_model_params, default_backdoor)
from utils.identifier_utils import mark_backdoors

from asr_testing import asr_eval
from mmlu_score import mmlu_score
from evaluate_perplexity import evaluate_perplexity


def main(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",
    clean_data_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/custom_data/clean_train.jsonl",
    poisoned_data_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/custom_data/poisoned_train.jsonl",
    output_dir: str = f"/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/checkpoints/",
    backdoor: str = "[TRIGGER]",
    front: bool = True,
    end: bool = False,
    loc: int = 0,
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 1,
    train_steps: int = 674,
    learning_rate: float = 1e-5,
    cutoff_len: int = 2048,
    # lora hyperparams
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Unlearning",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    prompt_template_name: str = "llama_chat",  # The prompt template to use, will default to alpaca.
    # additional data that can be added to the training/test set
    use_wandb: bool = True,
    seed: int = 11,
    clean_classification_accuracy: float = 1.0,
    poisoned_classification_accuracy: float = 0.0,
    eval_asr: bool = True,
    eval_mmlu: bool = True,
    eval_perplexity: bool = True,
    identify_backdoor: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"clean_data_path: {clean_data_path}\n"
            f"poisoned_data_path: {poisoned_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"train_steps: {train_steps}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"use_lora: {use_lora}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"prompt template: {prompt_template_name}\n"
            f"use_wandb: {use_wandb}\n"
            f"seed: {seed}\n"
            f"clean_classification_accuracy: {clean_classification_accuracy}\n"
            f"poisoned_classification_accuracy: {poisoned_classification_accuracy}\n"
            f"eval_asr: {eval_asr}\n"
            f"eval_mmlu: {eval_mmlu}\n"
            f"eval_perplexity: {eval_perplexity}\n"
            f"IDENTIFY_BACKDOOR: {identify_backdoor}\n"
        )
    if identify_backdoor:
        print("Warning!!! Identifying backdoors using the identifier module...")
    # assert backdoored_model_path is not None, "Please provide a backdoored model path"
    if not use_lora and learning_rate > 2e-5:
        print(
            "Warning: You are using a high learning rate without LoRA. This may cause instability."
        )
    backdoor_fn = lambda x: default_backdoor(x, backdoor, front, end, loc)
    gradient_accumulation_steps = batch_size // micro_batch_size
    file_name = f"""unlearn_identify_{identify_backdoor}_ca_{clean_classification_accuracy}_pa_{poisoned_classification_accuracy}_seed_{seed}_steps_{train_steps}_batch_{batch_size}"""
    output_dir = os.path.join(output_dir, file_name)
    wandb_run_name = file_name

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

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

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if clean_data_path.endswith(".json") or clean_data_path.endswith(".jsonl"):
        clean_data = load_dataset("json", data_files=clean_data_path)
    elif clean_data_path:
        clean_data = load_dataset(clean_data_path)

    if poisoned_data_path.endswith(".json") or poisoned_data_path.endswith(".jsonl"):
        poisoned_data = load_dataset("json", data_files=poisoned_data_path)
        # preappend the backdoor to the instructions in poisoned_data, the fields are instruction, input, output
    elif poisoned_data_path:
        poisoned_data = load_dataset(poisoned_data_path)
    if backdoor:
        poisoned_data['train'] = poisoned_data['train'].map(lambda x: {'instruction': backdoor_fn(x["instruction"]), 'input': x['input'], 'output': x['output']})

    if not identify_backdoor:
        column_names = poisoned_data["train"].column_names
        poisoned_data["train"] = poisoned_data["train"].map(generate_and_tokenize_prompt)
        clean_data["train"] = clean_data["train"].map(generate_and_tokenize_prompt)
        poisoned_data["train"] = poisoned_data["train"].remove_columns(column_names)
        clean_data["train"] = clean_data["train"].remove_columns(column_names)
        num_clean = len(clean_data["train"])
        num_correct = int(clean_classification_accuracy * num_clean)
        # -1 means backdoored, +1 means clean
        labels = np.ones(num_clean, dtype=int) * -1
        labels[:num_correct] = 1
        np.random.shuffle(labels)
        clean_data["train"].add_column("backdoor", labels)
        num_poisoned = len(poisoned_data["train"])
        num_correct = int(poisoned_classification_accuracy * num_poisoned)
        labels = np.ones(num_poisoned, dtype=int)
        labels[:num_correct] = -1
        np.random.shuffle(labels)
        poisoned_data["train"].add_column("backdoor", labels)
    else:
        print("Identifying backdoors...")
        # the data should have the actual backdoor label in the 'backdoor' column! change this
        clean_data['train'] = mark_backdoors(clean_data['train'], clean=True)
        poisoned_data['train'] = mark_backdoors(poisoned_data['train'], clean=False)
        # finish the mapping and column removal

    train_data = concatenate_datasets([poisoned_data['train'], clean_data['train']]).shuffle(seed=seed)
    print(train_data)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    
    if use_lora:
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if use_lora:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.\
    else:
        num_model_params = get_num_model_params(model)
        print(f"# model params: {num_model_params/1_000_000:.2f}M")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, train_steps: int, gradient_accumulation_steps: int,
            device: torch.device, checkpoint_file: str):
        pbar = None
        if is_main_proc():
            pbar = tqdm(total=train_steps)

        epoch = 0
        train_step = 0
        training_completed = False
        start_time = time.time()

        model.train()
        optimizer.zero_grad()

        while True:  # restart at the end of trainer
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                print(f"Setting sampler epoch: {epoch}")
                train_loader.sampler.set_epoch(epoch)

            for batch in train_loader:
                print(batch)
                tokenized_input = batch["input_ids"].to(device)
                label = batch["backdoor"].item()
                loss = model(input_ids=tokenized_input, labels=tokenized_input).loss
                loss = label * loss
                # loss = poisoned_loss - clean_loss - torch.pow(poisoned_loss - clean_loss, 2)
                # # clamp loss, and/or scale

                # Accumulate gradients
                loss.backward()

                if train_step % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                    optimizer.step()
                    optimizer.zero_grad()

                if pbar is not None:
                    pbar.set_description(f"Loss: {float(loss):.4f}")
                    pbar.update(1)
                if wandb.run is not None:
                    wandb.log({"train_loss": float(loss)})
                train_step += 1
                if train_step >= train_steps:
                    print(f"Training completed for {train_steps} steps. Stopping trainer.")
                    training_completed = True
                    break
            if training_completed:
                break
            epoch += 1

        time_elapsed_h = (time.time() - start_time) / (60 * 60)  # convert seconds into hours
        epochs_completed = train_step / len(train_loader)
        print(f"Model training finished / time elapsed: {time_elapsed_h:.2f}h / epochs completed: {epochs_completed:.2f} (counter: {epoch})")

        # Save the final checkpoint
        if is_main_proc() and checkpoint_file is not None:  # Save the final model
            if use_lora:
                model.save_pretrained(checkpoint_file)
            else:
                torch.save(model.state_dict(), checkpoint_file)
            print("Model state dict saved:", checkpoint_file)
    
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

    if use_wandb and is_main_proc():
        print("Initialization w&b...")
        wandb.init(project=wandb_project, name=wandb_run_name, resume=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = get_dataloader(train_data, micro_batch_size, tokenizer, 8, drop_last=False, generator=generator)
    optimizer = get_optimizer(model, lr=learning_rate, wd=0.0, maximize=True)

    # Train the model
    train(model, train_loader, optimizer, train_steps,
          gradient_accumulation_steps, device, checkpoint_file=output_dir)

    # wait_for_other_procs()
    print("!! Model training finished...")
    del optimizer

    if wandb.run is not None:
        wandb.finish()
    
    if eval_asr:
        asr_eval(model, tokenizer, run_name=wandb_run_name, backdoor=backdoor)

    if eval_perplexity:
        evaluate_perplexity(model, tokenizer, seed=seed, wandb_run_name=wandb_run_name, use_wandb=use_wandb)

    if eval_mmlu:
        mmlu_score(model, tokenizer)

if __name__ == "__main__":
    fire.Fire(main)
